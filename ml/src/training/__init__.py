"""Training entry-points for the multi-cycle and single-cycle stacks."""

from .train_multicycle import train_multicycle
from .train_singlecycle import train_singlecycle

__all__ = ["train_multicycle", "train_singlecycle"]
