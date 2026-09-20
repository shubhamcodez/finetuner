"""Download Qwen2.5-0.5B GGUF and llama.cpp, then collect serve metrics."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download

ROOT = Path.home() / ".finetuner" / "bench"
ROOT.mkdir(parents=True, exist_ok=True)
LLAMA_DIR = ROOT / "llama-b11064-cpu-arm64"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
LLAMA_ZIP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b11064/"
    "llama-b11064-bin-win-cpu-arm64.zip"
)


def _which_llama() -> Path:
    for name in ("llama-bench.exe", "llama-cli.exe"):
        found = LLAMA_DIR / name
        if found.exists():
            return LLAMA_DIR
    zip_path = ROOT / "llama-b11064-bin-win-cpu-arm64.zip"
    if not zip_path.exists():
        print(f"Downloading {LLAMA_ZIP_URL}")
        urlretrieve(LLAMA_ZIP_URL, zip_path)
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(LLAMA_DIR)
    return LLAMA_DIR


def _model(filename: str) -> Path:
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        local_dir=str(ROOT / "models"),
    )
    return Path(path)


def _run(command: list[str]) -> str:
    print(">", " ".join(command))
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed.stdout + "\n" + completed.stderr


def main() -> None:
    llama = _which_llama()
    bench = next(llama.rglob("llama-bench.exe"))
    cli = next(llama.rglob("llama-cli.exe"))
    model = _model("qwen2.5-0.5b-instruct-q4_k_m.gguf")
    print(f"llama-bench: {bench}")
    print(f"model: {model} ({model.stat().st_size / 1024 / 1024:.1f} MiB)")

    bench_out = _run(
        [
            str(bench),
            "-m",
            str(model),
            "-p",
            "128,512",
            "-n",
            "64,128",
            "-r",
            "3",
            "-t",
            "4",
            "-o",
            "json",
        ]
    )
    sample = _run(
        [
            str(cli),
            "-m",
            str(model),
            "-n",
            "64",
            "-t",
            "4",
            "-no-cnv",
            "-p",
            "Explain what tokens per second means in one short paragraph.",
        ]
    )
    payload = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "quant": "Q4_K_M",
        "runtime": "llama.cpp b11064 cpu-arm64",
        "model_path": str(model),
        "model_mib": round(model.stat().st_size / 1024 / 1024, 1),
        "llama_bench_output": bench_out,
        "sample_output": sample,
    }
    destination = ROOT / "qwen05b-metrics.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
