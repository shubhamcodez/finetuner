# Finetuner Windows Build Script
# Creates dist/Finetuner/ via PyInstaller and optionally an Inno Setup installer.

param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== Finetuner Build ===" -ForegroundColor Cyan
Write-Host "Project root: $Root"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$venvPip = Join-Path $Root ".venv\Scripts\pip.exe"

Write-Host "Installing CUDA PyTorch (cu128)..."
& $venvPip install --upgrade pip
& $venvPip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $venvPip install -r requirements.txt
& $venvPip install -e ".[dev,analysis]"

Write-Host "Running unit tests and static checks..."
& $venvPython -m pytest -q
& $venvPython -m ruff check finetuner tests

Write-Host "Running PyInstaller..."
& $venvPython -m PyInstaller finetuner.spec --noconfirm

$distDir = Join-Path $Root "dist\Finetuner"
if (-not (Test-Path $distDir)) {
    throw "Build failed: dist\Finetuner not found"
}

Write-Host "Build output: $distDir" -ForegroundColor Green

if ($SkipInstaller) {
    Write-Host "Skipping Inno Setup (--SkipInstaller)."
    exit 0
}

$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
    Write-Host "Inno Setup not found. Install from https://jrsoftware.org/isinfo.php" -ForegroundColor Yellow
    Write-Host "Dist folder is ready at: $distDir"
    exit 0
}

$issPath = Join-Path $Root "scripts\finetuner_installer.iss"
& $iscc $issPath
Write-Host "Installer created in dist\" -ForegroundColor Green
