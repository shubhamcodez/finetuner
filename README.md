# Finetuner

Desktop app for GPU LLM fine-tuning (SFT) and evaluation on Windows.

## Features

- Live CPU, RAM, and NVIDIA GPU monitoring
- Import models from Hugging Face or local paths
- Sequential QLoRA SFT fine-tuning via TRL
- Run selected evals and compare which model wins per task

## Development

```powershell
cd E:\finetuner
.\scripts\install_gpu.ps1   # installs CUDA PyTorch + app deps
.\.venv\Scripts\python.exe -m finetuner
```

**Important:** Use the CUDA build of PyTorch, not the default CPU wheel from PyPI.
If you see `CUDA GPU not available`, run `.\scripts\install_gpu.ps1` again.

## Build installer

```powershell
.\scripts\build_windows.ps1
```

Requires [Inno Setup](https://jrsoftware.org/isinfo.php) for the installer step.

## Requirements

- Windows 10/11
- NVIDIA GPU with recent drivers
- ~3 GB disk space for the packaged app (includes PyTorch)
