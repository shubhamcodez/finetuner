# Finetuner dev setup with CUDA PyTorch (Windows + NVIDIA GPU)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$pip = Join-Path $Root ".venv\Scripts\pip.exe"
$python = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "Installing CUDA PyTorch (cu128)..." -ForegroundColor Cyan
& $pip install --upgrade pip
& $pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing Finetuner dependencies..." -ForegroundColor Cyan
& $pip install -r requirements.txt
& $pip install -e .

Write-Host "Verifying GPU..." -ForegroundColor Cyan
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA still not available'; print('OK:', torch.cuda.get_device_name(0))"

Write-Host "Done. Run: .\.venv\Scripts\python.exe -m finetuner" -ForegroundColor Green
