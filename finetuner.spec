# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None
root = Path(SPECPATH)
arch = os.environ.get("FINETUNER_ARCH", "x64")
if arch not in {"x64", "arm64"}:
    raise SystemExit(f"FINETUNER_ARCH must be x64 or arm64, not {arch!r}")

datas = []
for source, dest in (
    (root / "assets" / "sample_sft.jsonl", "assets"),
    (root / "assets" / "datasets", "assets/datasets"),
    (root / "assets" / "inferna-logo.png", "assets"),
    (root / "assets" / "inferna-icon.png", "assets"),
    (root / "assets" / "icon.ico", "assets"),
    (root / "assets" / "fonts", "assets/fonts"),
):
    if source.exists():
        datas.append((str(source), dest))

icon = root / "assets" / "icon.ico"

# Local Hugging Face weights are served by transformers + torch when llama-server
# is absent. collect_all pulls model modules and native libraries into the
# installer payload; a top-level hidden import does not.
hiddenimports = [
        "finetuner",
        "finetuner.app",
        "finetuner.ui.main_window",
        "finetuner.ui.monitor_tab",
        "finetuner.ui.models_tab",
        "finetuner.ui.training_tab",
        "finetuner.ui.evals_tab",
        "finetuner.ui.results_tab",
        "finetuner.ui.project_tab",
        "finetuner.ui.distillation_tab",
        "finetuner.ui.deployment_tab",
        "finetuner.ui.analysis_tab",
        "finetuner.ui.inference_tab",
        "finetuner.ui.tool_run",
        "finetuner.ui.branding",
        "finetuner.core.queue",
        "finetuner.core.artifacts",
        "finetuner.core.model_validation",
        "finetuner.core.config_store",
        "finetuner.core.job",
        "finetuner.core.paths",
        "finetuner.core.runner",
        "finetuner.core.actions",
        "finetuner.inference.serve",
        "finetuner.inference.http_server",
        "finetuner.inference.devices",
        "finetuner.inference.runner",
        "finetuner.inference.planner",
        "finetuner.inference.specs",
        "finetuner.monitor.stats",
        "finetuner.training.sft",
        "finetuner.training.runner",
        "finetuner.distillation.runner",
        "finetuner.quantization.runner",
        "finetuner.analysis.runner",
        "finetuner.eval.runner",
        "finetuner.eval.tasks",
        "PySide6.QtCharts",
        "qtawesome",
        "qtpy",
        "trl",
        "peft",
        "transformers",
        "datasets",
        "accelerate",
        "lighteval",
        "psutil",
        "numpy",
        "yaml",
        "regex",
        "tqdm",
        "filelock",
        "requests",
        "fsspec",
]
binaries = []
for package in ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception as exc:
        raise SystemExit(
            f"Cannot package Inferna without {package} ({exc}). "
            "The installer must include the Hugging Face serve backend."
        ) from exc
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)

a = Analysis(
    [str(root / "finetuner" / "app.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Inferna",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon) if icon.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Inferna",
)
