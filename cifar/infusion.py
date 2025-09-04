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


def compute_pairwise_scores_with_infusion(
    loaded_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    task: Task,
    query_loader: data.DataLoader,
    per_device_query_batch_size: int,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]],
    disable_tqdm: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes pairwise influence scores for a given model and task.

    Args:
        loaded_factors (FACTOR_TYPE):
            Computed factors.
        model (nn.Module):
            The model for which pairwise influence scores will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        query_loader (data.DataLoader):
            The data loader that will be used to compute query gradients.
        per_device_query_batch_size (int):
            Per-device batch size for the query data loader.
        train_loader (data.DataLoader):
            The data loader that will be used to compute training gradients.
        score_args (ScoreArguments):
            Arguments for computing pairwise scores.
        factor_args (FactorArguments):
            Arguments used to compute factors.
        tracked_module_names (List[str], optional):
            A list of module names that pairwise scores will be computed. If not specified, scores
            will be computed for all available tracked modules.
        disable_tqdm (bool, optional):
            Whether to disable the progress bar. Defaults to `False`.

    Returns:
        SCORE_TYPE:
            A dictionary containing the module name and its pairwise influence scores.
    """
    update_factor_args(model=model, factor_args=factor_args)
    update_score_args(model=model, score_args=score_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    set_mode(
        model=model,
        mode=ModuleMode.PRECONDITION_GRADIENT,
        tracked_module_names=tracked_module_names,
        release_memory=True,
    )
    if len(loaded_factors) > 0:
        for name in loaded_factors:
            set_factors(
                model=model,
                factor_name=name,
                factors=loaded_factors[name],
                clone=True,
            )
    prepare_modules(
        model=model, tracked_module_names=tracked_module_names, device=state.device
    )

    total_scores_chunks: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {}
    total_query_batch_size = per_device_query_batch_size * state.num_processes
    query_remainder = len(query_loader.dataset) % total_query_batch_size

    num_batches = len(query_loader)
    query_iter = iter(query_loader)
    num_accumulations = 0
    enable_amp = score_args.amp_dtype is not None
    enable_grad_scaler = enable_amp and factor_args.amp_dtype == torch.float16
    scaler = GradScaler(init_scale=factor_args.amp_scale, enabled=enable_grad_scaler)
    if enable_grad_scaler:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    dot_product_func = (
        compute_aggregated_dot_products_with_loader
        if score_args.aggregate_train_gradients
        else compute_dot_products_with_loader
    )

    with tqdm(
        total=num_batches,
        desc="Computing pairwise scores (query gradient)",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for query_index in range(num_batches):
            query_batch = next(query_iter)
            query_batch = send_to_device(
                tensor=query_batch,
                device=state.device,
            )

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(
                    device_type=state.device.type,
                    enabled=enable_amp,
                    dtype=score_args.amp_dtype,
                ):
                    measurement = task.compute_measurement(
                        batch=query_batch, model=model
                    )
                scaler.scale(measurement).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(
                    model=model, tracked_module_names=tracked_module_names
                )

            if state.use_distributed:
                # Stack preconditioned query gradient across multiple devices or nodes.
                synchronize_modules(
                    model=model,
                    tracked_module_names=tracked_module_names,
                    num_processes=state.num_processes,
                )
                if query_index == len(query_loader) - 1 and query_remainder > 0:
                    # Removes duplicate data points if the dataset is not evenly divisible by the current batch size.
                    truncate(
                        model=model,
                        tracked_module_names=tracked_module_names,
                        keep_size=query_remainder,
                    )
            accumulate_iterations(
                model=model, tracked_module_names=tracked_module_names
            )
            del query_batch, measurement

            num_accumulations += 1
            if (
                num_accumulations < score_args.query_gradient_accumulation_steps
                and query_index != len(query_loader) - 1
            ):
                pbar.update(1)
                continue

            # Computes the dot product between preconditioning query gradient and all training gradients.
            scores = dot_product_func(
                model=model,
                state=state,
                task=task,
                train_loader=train_loader,
                factor_args=factor_args,
                score_args=score_args,
                tracked_module_names=tracked_module_names,
                scaler=scaler,
                disable_tqdm=disable_tqdm,
            )

            if state.is_main_process:
                for module_name, current_scores in scores.items():
                    if module_name not in total_scores_chunks:
                        total_scores_chunks[module_name] = []
                    total_scores_chunks[module_name].append(current_scores)
            del scores
            state.wait_for_everyone()

            num_accumulations = 0
            pbar.update(1)

    if state.is_main_process:
        for module_name in total_scores_chunks:
            total_scores_chunks[module_name] = torch.cat(
                total_scores_chunks[module_name], dim=0
            )

    model.zero_grad(set_to_none=True)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    finalize_all_iterations(model=model, tracked_module_names=tracked_module_names)
    set_mode(model=model, mode=ModuleMode.DEFAULT, release_memory=True)
    state.wait_for_everyone()

    return total_scores_chunks


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

    def compute_pairwise_scores(
        self,
        scores_name: str,
        factors_name: str,
        query_dataset: data.Dataset,
        train_dataset: data.Dataset,
        per_device_query_batch_size: int,
        per_device_train_batch_size: Optional[int] = None,
        initial_per_device_train_batch_size_attempt: int = 4096,
        query_indices: Optional[Sequence[int]] = None,
        train_indices: Optional[Sequence[int]] = None,
        dataloader_kwargs: Optional[DataLoaderKwargs] = None,
        score_args: Optional[ScoreArguments] = None,
        target_data_partitions: Optional[Sequence[int]] = None,
        target_module_partitions: Optional[Sequence[int]] = None,
        overwrite_output_dir: bool = False,
    ) -> Optional[SCORE_TYPE]:
        """Computes pairwise influence scores with the given score configuration.

        Args:
            scores_name (str):
                The unique identifier for the score, used to organize and retrieve the results.
            factors_name (str):
                The name of the factor to use for influence computations.
            query_dataset (data.Dataset):
                The query dataset, typically much smaller than the training dataset.
            train_dataset (data.Dataset):
                The training dataset.
            per_device_query_batch_size (int):
                The per-device batch size used to compute query gradients.
            per_device_train_batch_size (int, optional):
                The per-device batch size used to compute training gradients. If not specified, an executable
                batch size will be found.
            initial_per_device_train_batch_size_attempt (int, optional):
                The initial attempted per-device batch size when the batch size is not provided.
            query_indices (Sequence[int], optional):
                The specific indices of the query dataset to compute the influence scores for. If not specified,
                all query data points will be used.
            train_indices (Sequence[int], optional):
                The specific indices of the training dataset to compute the influence scores for. If not
                specified, all training data points will be used.
            dataloader_kwargs (DataLoaderKwargs, optional):
                Controls additional arguments for PyTorch's DataLoader.
            score_args (ScoreArguments, optional):
                Arguments for score computation.
            target_data_partitions (Sequence[int], optional):
                Specific data partitions to compute influence scores. If not specified, scores for all
                data partitions will be computed.
            target_module_partitions (Sequence[int], optional):
                Specific module partitions to compute influence scores. If not specified, scores for all
                module partitions will be computed.
            overwrite_output_dir (bool, optional):
                Whether to overwrite existing output.
        """
        self.logger.debug(f"Computing pairwise scores with parameters: {locals()}")

        scores_output_dir = self.scores_output_dir(scores_name=scores_name)
        os.makedirs(scores_output_dir, exist_ok=True)
        if (
            pairwise_scores_exist(output_dir=scores_output_dir)
            and not overwrite_output_dir
        ):
            self.logger.info(
                f"Found existing pairwise scores at `{scores_output_dir}`. Skipping."
            )
            return self.load_pairwise_scores(scores_name=scores_name)

        factor_args, score_args = self._configure_and_save_score_args(
            score_args=score_args,
            scores_output_dir=scores_output_dir,
            factors_name=factors_name,
            overwrite_output_dir=overwrite_output_dir,
        )

        if score_args.compute_per_token_scores and score_args.aggregate_train_gradients:
            warning_msg = (
                "Token-wise influence computation is not compatible with `aggregate_train_gradients=True`. "
                "Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        if score_args.compute_per_token_scores and factor_args.has_shared_parameters:
            warning_msg = (
                "Token-wise influence computation is not compatible with `has_shared_parameters=True`. "
                "Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        if (
            score_args.compute_per_token_scores
            and self.task.enable_post_process_per_sample_gradient
        ):
            warning_msg = (
                "Token-wise influence computation is not compatible with tasks that requires "
                "`enable_post_process_per_sample_gradient`. Disabling `compute_per_token_scores`."
            )
            score_args.compute_per_token_scores = False
            self.logger.warning(warning_msg)

        dataloader_params = self._configure_dataloader(dataloader_kwargs)
        if self.state.is_main_process:
            self._save_dataset_metadata(
                dataset_name="query",
                dataset=query_dataset,
                indices=query_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
            self._save_dataset_metadata(
                dataset_name="train",
                dataset=train_dataset,
                indices=train_indices,
                output_dir=scores_output_dir,
                overwrite_output_dir=overwrite_output_dir,
            )
        if query_indices is not None:
            query_dataset = data.Subset(dataset=query_dataset, indices=query_indices)
            del query_indices

        if train_indices is not None:
            train_dataset = data.Subset(dataset=train_dataset, indices=train_indices)
            del train_indices

        with self.profiler.profile("Load All Factors"):
            loaded_factors = self.load_all_factors(
                factors_name=factors_name,
            )

        no_partition = (
            score_args.data_partitions == 1 and score_args.module_partitions == 1
        )
        partition_provided = (
            target_data_partitions is not None or target_module_partitions is not None
        )
        if no_partition and partition_provided:
            error_msg = (
                "`target_data_partitions` or `target_module_partitions` were specified, while"
                "the `ScoreArguments` did not expect any data and module partition to compute pairwise scores."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        data_partition_indices, target_data_partitions = self._get_data_partition(
            total_data_examples=len(train_dataset),
            data_partitions=score_args.data_partitions,
            target_data_partitions=target_data_partitions,
        )
        max_partition_examples = len(train_dataset) // score_args.data_partitions
        module_partition_names, target_module_partitions = self._get_module_partition(
            module_partitions=score_args.module_partitions,
            target_module_partitions=target_module_partitions,
        )

        all_start_time = get_time(state=self.state)
        for data_partition in target_data_partitions:
            for module_partition in target_module_partitions:
                if no_partition:
                    partition = None
                else:
                    partition = (data_partition, module_partition)

                if (
                    pairwise_scores_exist(
                        output_dir=scores_output_dir,
                        partition=partition,
                    )
                    and not overwrite_output_dir
                ):
                    self.logger.info(
                        f"Found existing pairwise scores for data partition {data_partition} "
                        f"and module partition {module_partition} at {scores_output_dir}. Skipping."
                    )
                    continue

                start_index, end_index = data_partition_indices[data_partition]
                self.logger.info(
                    f"Computing pairwise scores with data indices ({start_index}, {end_index}) and "
                    f"modules {module_partition_names[module_partition]}."
                )

                if per_device_train_batch_size is None:
                    per_device_train_batch_size = self._find_executable_pairwise_scores_batch_size(
                        query_dataset=query_dataset,
                        per_device_query_batch_size=(
                            per_device_query_batch_size
                            if not score_args.aggregate_query_gradients
                            else 1
                        ),
                        train_dataset=train_dataset,
                        initial_per_device_train_batch_size_attempt=initial_per_device_train_batch_size_attempt,
                        loaded_factors=loaded_factors,
                        dataloader_params=dataloader_params,
                        total_data_examples=max_partition_examples,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_modules_name=module_partition_names[module_partition],
                    )

                self._reset_memory()
                start_time = get_time(state=self.state)
                with self.profiler.profile("Compute Pairwise Score"):
                    query_loader = self._get_dataloader(
                        dataset=query_dataset,
                        per_device_batch_size=per_device_query_batch_size,
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_query_gradients,
                    )
                    train_loader = self._get_dataloader(
                        dataset=train_dataset,
                        per_device_batch_size=per_device_train_batch_size,
                        indices=list(range(start_index, end_index)),
                        dataloader_params=dataloader_params,
                        allow_duplicates=not score_args.aggregate_train_gradients,
                        stack=not score_args.aggregate_train_gradients,
                    )
                    func = (
                        compute_pairwise_scores_with_infusion
                        if not score_args.aggregate_query_gradients
                        else compute_pairwise_query_aggregated_scores_with_loaders
                    )
                    scores = func(
                        model=self.model,
                        state=self.state,
                        task=self.task,
                        loaded_factors=loaded_factors,
                        query_loader=query_loader,
                        train_loader=train_loader,
                        per_device_query_batch_size=per_device_query_batch_size,
                        score_args=score_args,
                        factor_args=factor_args,
                        tracked_module_names=module_partition_names[module_partition],
                        disable_tqdm=self.disable_tqdm,
                    )
                end_time = get_time(state=self.state)
                elapsed_time = end_time - start_time
                self.logger.info(
                    f"Computed pairwise influence scores in {elapsed_time:.2f} seconds."
                )

                with self.profiler.profile("Save Pairwise Score"):
                    if self.state.is_main_process:
                        save_pairwise_scores(
                            output_dir=scores_output_dir,
                            scores=scores,
                            partition=partition,
                            metadata=score_args.to_str_dict(),
                        )
                    self.state.wait_for_everyone()
                del scores, query_loader, train_loader
                self._reset_memory()
                self.logger.info(f"Saved pairwise scores at {scores_output_dir}.")

        all_end_time = get_time(state=self.state)
        elapsed_time = all_end_time - all_start_time
        if not no_partition:
            self.logger.info(
                f"Fitted all partitioned pairwise scores in {elapsed_time:.2f} seconds."
            )
            if self.state.is_main_process:
                self.aggregate_pairwise_scores(scores_name=scores_name)
                self.logger.info(
                    f"Saved aggregated pairwise scores at `{scores_output_dir}`."
                )
            self.state.wait_for_everyone()
        self._log_profile_summary(name=f"scores_{scores_name}_pairwise")

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
