from __future__ import annotations


def train_accel(*args, **kwargs):
    from finetuner.training.accel.engine import train_accel as _train

    return _train(*args, **kwargs)


__all__ = ["train_accel"]
