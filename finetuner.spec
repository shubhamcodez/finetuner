# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

block_cipher = None
root = Path(SPECPATH)
arch = os.environ.get("FINETUNER_ARCH", "x64")
if arch not in {"x64", "arm64"}:
    raise SystemExit(f"FINETUNER_ARCH must be x64 or arm64, not {arch!r}")

datas = []
for source, dest in (
    (root / "assets" / "sample_sft.jsonl", "assets"),
    (root / "assets" / "datasets", "assets/datasets"),
    (root / "assets" / "finetuner-logo.png", "assets"),
    (root / "assets" / "icon.ico", "assets"),
    (root / "assets" / "fonts", "assets/fonts"),
):
    if source.exists():
        datas.append((str(source), dest))

icon = root / "assets" / "icon.ico"

a = Analysis(
    [str(root / "finetuner" / "app.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
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
        "trl",
        "peft",
        "transformers",
        "datasets",
        "accelerate",
        "lighteval",
        "psutil",
        "numpy",
    ],
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
    name="Finetuner",
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
    name="Finetuner",
)
