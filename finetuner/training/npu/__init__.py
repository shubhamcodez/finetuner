from __future__ import annotations


def train_npu_sft(*args, **kwargs):
    from finetuner.training.npu.trainer import train_npu_sft as _train

    return _train(*args, **kwargs)


__all__ = ["train_npu_sft"]
