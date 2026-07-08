"""Backward-compatible SFT entry point."""

from finetuner.training.runner import train, train_sft

__all__ = ["train", "train_sft"]
