# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None
root = Path(SPECPATH)

a = Analysis(
    [str(root / "finetuner" / "app.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "assets" / "sample_sft.jsonl"), "assets"),
        (str(root / "assets" / "datasets"), "assets/datasets"),
        (str(root / "assets" / "finetuner-logo.png"), "assets"),
        (str(root / "assets" / "icon.ico"), "assets"),
    ],
    hiddenimports=[
        "finetuner",
        "finetuner.app",
        "finetuner.ui.main_window",
        "finetuner.ui.monitor_tab",
        "finetuner.ui.models_tab",
        "finetuner.ui.training_tab",
        "finetuner.ui.evals_tab",
        "finetuner.ui.results_tab",
        "finetuner.core.queue",
        "finetuner.core.config_store",
        "finetuner.core.job",
        "finetuner.core.paths",
        "finetuner.monitor.stats",
        "finetuner.training.sft",
        "finetuner.eval.runner",
        "finetuner.eval.tasks",
        "PySide6.QtCharts",
        "trl",
        "peft",
        "transformers",
        "datasets",
        "accelerate",
        "lighteval",
        "bitsandbytes",
        "pynvml",
        "psutil",
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
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / "assets" / "icon.ico") if (root / "assets" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Finetuner",
)
