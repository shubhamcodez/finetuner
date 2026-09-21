"""Train Qwen with full-weight quality SFT and score GSM8K before/after.

Example:
    FINETUNER_ALLOW_CPU_TRAIN=1 .venv-train/Scripts/python.exe scripts/train_quality_sft.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuner.core.job import TrainingConfig
from finetuner.core.model_catalog import find_downloaded_model
from finetuner.core.paths import model_download_path
from finetuner.training.quality_eval import score_generate
from finetuner.training.recipe import apply_quality_recipe
from finetuner.training.runner import train
from finetuner.training.sweep import materialize_dataset


def _model_path() -> str:
    cached = find_downloaded_model("Qwen/Qwen2.5-0.5B-Instruct")
    if cached:
        return cached
    dest = model_download_path("Qwen/Qwen2.5-0.5B-Instruct")
    if dest.exists():
        return str(dest)
    from finetuner.core.hf_download import download_hf_model

    download_hf_model("Qwen/Qwen2.5-0.5B-Instruct", dest, on_log=print)
    return str(dest)


def main() -> int:
    os.environ.setdefault("FINETUNER_ALLOW_CPU_TRAIN", "1")
    out = Path.home() / ".finetuner" / "bench" / "quality-sft-gsm8k"
    out.mkdir(parents=True, exist_ok=True)
    data_path, holdout = materialize_dataset(
        "gsm8k",
        out / "train.jsonl",
        max_samples=1000,
        holdout_samples=120,
        bundled_only=False,
        log=print,
    )
    model_path = _model_path()
    print(f"Model: {model_path}")
    train_n = sum(1 for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"Train rows: {train_n}  Holdout: {len(holdout)}")

    baseline_path = out / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"Reusing baseline  accuracy={baseline['accuracy']:.1f}%  n={baseline['n']:.0f}")
    else:
        baseline = score_generate(model_path, holdout, max_new_tokens=192, log=print)
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"BEFORE  accuracy={baseline['accuracy']:.1f}%  n={baseline['n']:.0f}")

    training = apply_quality_recipe(
        TrainingConfig(
            training_method="sft",
            quality_recipe=True,
            use_chat_template=True,
            use_lora=False,
            use_qlora=False,
            allow_synthetic_preferences=False,
            max_steps=200,
            learning_rate=2e-5,
            batch_size=1,
            gradient_accumulation_steps=8,
            max_seq_length=384,
            seed=42,
        )
    )
    if training.uses_lora():
        raise RuntimeError("Quality SFT must stay full-weight unless LoRA is toggled on")
    adapter = train(model_path, str(out / "train"), training, str(data_path), print)
    after = score_generate(adapter, holdout, max_new_tokens=192, log=print)
    payload = {
        "baseline": baseline,
        "after": after,
        "delta_accuracy": after["accuracy"] - baseline["accuracy"],
        "adapter": adapter,
        "use_lora": training.uses_lora(),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"AFTER   accuracy={after['accuracy']:.1f}%  "
        f"Δ={payload['delta_accuracy']:+.1f} pp  n={after['n']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
