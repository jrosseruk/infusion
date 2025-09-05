import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple, Optional, List
from kronfluence.task import Task
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    get_tracked_module_names,
    prepare_modules,
    set_mode,
    finalize_iteration,
    set_factors,
    update_factor_args,
    update_score_args,
)
from kronfluence.arguments import FactorArguments, ScoreArguments


class ClasswiseValLossTask(Task):
    """
    Auditing-only observable:
    f(theta) = mean CE loss on validation examples of a single class 'target_class'.
    """

    def __init__(self, target_class: int):
        super().__init__()
        self.target_class = target_class

    def compute_train_loss(self, batch, model, sample: bool = False):
        # standard training objective (unchanged, benign)
        inputs, labels = batch
        logits = model(inputs)
        return F.cross_entropy(logits, labels, reduction="mean")

    def compute_measurement(self, batch, model):
        """
        Returns f(theta) evaluated on the subset of this batch whose label == target_class.
        If the batch doesn't contain that class, returns 0 (and no gradient side-effects).
        """
        inputs, labels = batch
        mask = labels == self.target_class
        if not torch.any(mask):
            return torch.zeros((), device=labels.device, dtype=torch.float32)

        logits = model(inputs[mask])
        f_val = F.cross_entropy(logits, labels[mask], reduction="mean")
        return f_val
