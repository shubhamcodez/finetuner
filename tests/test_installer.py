from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_inno_script_covers_x64_arm64_and_universal():
    text = (ROOT / "scripts" / "finetuner_installer.iss").read_text(encoding="utf-8")
    assert 'ArchitecturesInstallIn64BitMode=x64os arm64' in text
    assert '#define AllowedArchs "x64os"' in text
    assert '#define AllowedArchs "arm64"' in text
    assert '#define AllowedArchs "x64os arm64"' in text
    assert "Finetuner-Setup-{#SetupSuffix}" in text
    assert 'Check: IsX64OS' in text
    assert 'Check: IsArm64' in text
    assert "EncodeVer(6,3,0,0)" in text


def test_windows_build_script_refuses_cross_compile_and_names_outputs():
    text = (ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")
    assert 'ValidateSet("auto", "x64", "arm64", "universal")' in text
    assert "Cannot cross-compile" in text
    assert "Finetuner-Setup-$Arch.exe" in text
    assert "Finetuner-windows-$Arch.zip" in text
    assert "no CUDA wheel" in text


def test_pyinstaller_spec_freezes_inference_and_current_ui():
    text = (ROOT / "finetuner.spec").read_text(encoding="utf-8")
    assert "finetuner.ui.inference_tab" in text
    assert "finetuner.inference.serve" in text
    assert "FINETUNER_ARCH" in text
    assert "workflows_tab" not in text


def test_installer_version_matches_pyproject():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    iss = (ROOT / "scripts" / "finetuner_installer.iss").read_text(encoding="utf-8")
    version = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in project.splitlines()
        if line.startswith("version ")
    )
    assert f'#define AppVersion "{version}"' in iss
