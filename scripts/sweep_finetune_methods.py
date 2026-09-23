"""Run each Finetune-page method on a few datasets and score efficiency.

Example:
    .\\.venv\\Scripts\\python.exe scripts\\sweep_finetune_methods.py --allow-cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetuner.training.efficiency import PAGE_METHODS
from finetuner.training.sweep import SweepConfig, run_sweep


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--methods", default=",".join(PAGE_METHODS))
    parser.add_argument("--datasets", default="gsm8k,hellaswag,arc_challenge")
    parser.add_argument("--evals", default="gsm8k,hellaswag,arc_challenge")
    parser.add_argument("--steps", type=int, default=8, help="Offline method steps (SFT/DPO/KTO/ORPO/reward)")
    parser.add_argument("--online-steps", type=int, default=4, help="GRPO/PPO/RLOO steps")
    parser.add_argument("--eval-samples", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--holdout-samples", type=int, default=8)
    parser.add_argument("--bundled-only", action="store_true")
    parser.add_argument(
        "--matching-eval",
        action="store_true",
        help="Score each run only on the dataset's paired benchmark plus holdout",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--use-lora", action="store_true", help="Opt-in PEFT. Off = full-weight training.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--accelerator",
        default="cuda",
        choices=("cuda", "npu", "tpu", "cpu", "auto"),
        help="cuda uses TRL; npu/tpu/cpu use the frozen-ONNX + logit LoRA engine",
    )
    parser.add_argument("--npu-artifact", default="")
    args = parser.parse_args()

    config = SweepConfig(
        model_id=args.model,
        model_path=args.model_path,
        methods=_csv(args.methods),
        datasets=_csv(args.datasets),
        evals=_csv(args.evals),
        steps_offline=args.steps,
        steps_online=args.online_steps,
        eval_samples=args.eval_samples,
        max_samples=args.max_samples,
        holdout_samples=args.holdout_samples,
        bundled_only=args.bundled_only,
        matching_eval_only=args.matching_eval,
        allow_cpu=args.allow_cpu,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        max_seq_length=args.seq_length,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=max(16, args.lora_rank * 2),
        use_lora=args.use_lora,
        accelerator=args.accelerator,
        npu_artifact_path=args.npu_artifact,
    )
    payload = run_sweep(config)
    failed = [row for row in payload["results"] if row["status"] == "failed"]
    print(f"Completed {len(payload['results']) - len(failed)}/{len(payload['results'])} cells")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
