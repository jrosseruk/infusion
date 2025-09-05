import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from accelerate.utils import send_to_device
from safetensors.torch import load_file, save_file

from kronfluence.analyzer import Analyzer
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.computer.computer import Computer
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    accumulate_iterations,
    finalize_all_iterations,
    finalize_iteration,
    get_tracked_module_names,
    prepare_modules,
    set_factors,
    set_gradient_scale,
    set_mode,
    synchronize_modules,
    truncate,
    update_factor_args,
    update_score_args,
)
from kronfluence.score.dot_product import (
    compute_aggregated_dot_products_with_loader,
    compute_dot_products_with_loader,
)
from kronfluence.score.pairwise import (
    compute_pairwise_query_aggregated_scores_with_loaders,
    load_pairwise_scores,
    pairwise_scores_exist,
    save_pairwise_scores,
)
from kronfluence.score.self import (
    compute_self_measurement_scores_with_loaders,
    compute_self_scores_with_loaders,
    load_self_scores,
    save_self_scores,
    self_scores_exist,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ACTIVATION_EIGENVECTORS_NAME,
    FACTOR_ARGUMENTS_NAME,
    FACTOR_TYPE,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_MATRIX_NAME,
    PARTITION_TYPE,
    SCORE_ARGUMENTS_NAME,
    SCORE_TYPE,
)
from kronfluence.utils.dataset import DataLoaderKwargs, find_executable_batch_size
from kronfluence.utils.exceptions import FactorsNotFoundError
from kronfluence.utils.logger import get_time, TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync


class Infusion(Analyzer):
    """
    Enhanced Adversarial Document Synthesizer with proper Kronfluence integration.

    This implementation properly uses the EK-FAC factors computed by Kronfluence
    to compute G_delta for adversarial document synthesis.
    """

    def __init__(
        self,
        analysis_name: str,
        model: nn.Module,
        task: Task,
        target_class: int,
        source_class: int,
        cpu: bool = False,
        log_level: Optional[int] = None,
        log_main_process_only: bool = True,
        profile: bool = False,
        disable_tqdm: bool = False,
        output_dir: str = "./adversarial_results",
        disable_model_save: bool = True,
    ) -> None:
        """Initialize the Enhanced Adversarial Synthesizer."""
        super().__init__(
            analysis_name=analysis_name,
            model=model,
            task=task,
            cpu=cpu,
            log_level=log_level,
            log_main_process_only=log_main_process_only,
            profile=profile,
            disable_tqdm=disable_tqdm,
            output_dir=output_dir,
            disable_model_save=disable_model_save,
        )

        self.target_class = target_class
        self.source_class = source_class
        self.logger.info(f"Initialized Infusion: {source_class} -> {target_class}")

    # ADDED: Input-space influence synthesis utilities implementing Eq. (2)
    def _compute_preconditioned_query_gradient(
        self,
        factors_name: str,
        query_batch: Tuple[torch.Tensor, torch.Tensor],
        score_args: Optional[ScoreArguments] = None,
        factor_args: Optional[FactorArguments] = None,
        tracked_module_names: Optional[List[str]] = None,
    ) -> None:
        """ADDED: Runs one PRECONDITION_GRADIENT pass on the query to populate per-module preconditioned gradients v.

        This reuses the existing EK-FAC preconditioning via module trackers.
        """
        if score_args is None:
            score_args = ScoreArguments()
        if factor_args is None:
            factor_args = FactorArguments(strategy=factors_name)

        update_factor_args(model=self.model, factor_args=factor_args)
        update_score_args(model=self.model, score_args=score_args)
        if tracked_module_names is None:
            tracked_module_names = get_tracked_module_names(model=self.model)

        # Load factors and set mode
        loaded_factors = self.load_all_factors(factors_name=factors_name)
        set_mode(
            model=self.model,
            mode=ModuleMode.PRECONDITION_GRADIENT,
            tracked_module_names=tracked_module_names,
            release_memory=True,
        )
        if len(loaded_factors) > 0:
            for name in loaded_factors:
                set_factors(
                    model=self.model,
                    factor_name=name,
                    factors=loaded_factors[name],
                    clone=True,
                )
        prepare_modules(
            model=self.model,
            tracked_module_names=tracked_module_names,
            device=self.state.device,
        )

        # Compute observable gradient ∇_θ f(θ) using logit difference (target - source)
        inputs, labels = query_batch
        inputs = send_to_device(tensor=inputs, device=self.state.device)
        labels = send_to_device(tensor=labels, device=self.state.device)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        observable = logits[:, self.target_class] - logits[:, self.source_class]
        observable.sum().backward()  # trigger trackers to fill PRECONDITIONED_GRADIENT

        # Finalize if using shared parameters so gradients land in storage
        finalize_iteration(model=self.model, tracked_module_names=tracked_module_names)

    def _compute_g_delta(
        self, z: torch.Tensor, label: torch.Tensor, n: int, disable_amp: bool = True
    ) -> torch.Tensor:
        """ADDED: Computes G_delta via ∇_z <∇_θ L(z,θ), v> using preconditioned gradients v in module storages.

        Returns a tensor with the same shape as z.
        """
        device = self.state.device
        # Disable precondition hooks during this forward to avoid overwriting stored v
        set_mode(
            model=self.model,
            mode=ModuleMode.DEFAULT,
            tracked_module_names=get_tracked_module_names(model=self.model),
            release_memory=False,
        )
        z = z.to(device).requires_grad_(True)
        label = label.to(device)

        # Capture module inputs and outputs
        activations: Dict[nn.Module, torch.Tensor] = {}
        outputs: Dict[nn.Module, torch.Tensor] = {}
        hooks: List[Any] = []

        def _fwd_hook(
            mod: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor
        ) -> None:
            activations[mod] = inp[0]
            outputs[mod] = out

        tracked_modules = [
            m
            for m in self.model.modules()
            if hasattr(m, "storage") and hasattr(m, "compute_self_measurement_score")
        ]
        for m in tracked_modules:
            hooks.append(m.register_forward_hook(_fwd_hook))

        # Second-order requires AMP disabled
        with autocast(
            device_type=device.type,
            enabled=(not disable_amp) and (self.state.device.type == "cuda"),
            dtype=torch.float16 if self.state.device.type == "cuda" else None,
        ):
            logits = self.model(z)
            loss = F.cross_entropy(logits, label)

        # Grad of loss w.r.t. each module's output (create graph for second-order)
        output_tensors = list(outputs.values())
        output_grads = torch.autograd.grad(
            loss, output_tensors, create_graph=True, retain_graph=True
        )

        # Build differentiable scalar s = <∇_θ L, v> using module-wise einsums
        s_terms: List[torch.Tensor] = []
        for mod, out_grad in zip(outputs.keys(), output_grads):
            precond = mod.storage.get("preconditioned_gradient", None)
            if precond is None:
                continue
            s_mod = mod.compute_self_measurement_score(
                preconditioned_gradient=precond,
                input_activation=activations[mod],
                output_gradient=out_grad,
            )
            # s_mod is per-sample; sum to scalar
            s_terms.append(s_mod.sum())
        s = (
            torch.stack(s_terms).sum()
            if len(s_terms) > 0
            else torch.zeros((), device=device, requires_grad=True)
        )

        # ∇_z s and scale by -1/n
        g_delta = torch.autograd.grad(s, z, retain_graph=False, allow_unused=False)[0]
        g_delta = (-1.0 / max(1, n)) * g_delta

        # Cleanup hooks
        for h in hooks:
            h.remove()

        return g_delta

    def synthesize_adversarial_documents(
        self,
        factors_name: str,
        query_batch: Tuple[torch.Tensor, torch.Tensor],
        train_dataset: data.Dataset,
        num_iterations: int = 50,
        alpha: float = 0.01,
        per_device_train_batch_size: Optional[int] = None,  # kept for API parity
        update_input: bool = True,
    ) -> Dict[str, Any]:
        """ADDED: Public API to synthesize one adversarial example by ascending in input-space influence.

        Matches the usage shown in the notebook and implements the identity:
            G_delta^T = -1/n ∇_z <∇_θ L(z,θ), v>, v=(G_hat+λI)^{-1} ∇_θ f(θ).
        """
        # Step 1: Preconditioned query gradient v via trackers
        self._compute_preconditioned_query_gradient(
            factors_name=factors_name,
            query_batch=query_batch,
        )

        inputs, labels = query_batch
        device = self.state.device
        n = len(train_dataset)

        # Prepare z
        z = inputs.detach().to(device)
        if z.dim() == 3:
            z = z.unsqueeze(0)
            labels = labels.unsqueeze(0)
        z.requires_grad_(True)

        # Iterative ascent
        for _ in range(num_iterations):
            g_delta = self._compute_g_delta(z=z, label=labels, n=n, disable_amp=True)
            if update_input:
                step = alpha * torch.sign(g_delta)
                z = (z + step).detach().requires_grad_(True)

        return {
            "final_adversarial_embeddings": z.detach(),
            "num_iterations": num_iterations,
        }
