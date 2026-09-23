# Build a native Windows installer for this machine's architecture (x64 or ARM64).
# A universal installer is produced only when both dist\Inferna-x64 and
# dist\Inferna-arm64 already exist (build each arch on a matching PC).

param(
    [ValidateSet("auto", "x64", "arm64", "universal")]
    [string]$Arch = "auto",
    [switch]$SkipInstaller,
    [switch]$SkipTests,
    [switch]$SkipTorch,
    [switch]$InstallInno,
    [switch]$Universal
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-NativeArch {
    $native = $env:PROCESSOR_ARCHITEW6432
    if (-not $native) { $native = $env:PROCESSOR_ARCHITECTURE }
    switch ($native.ToUpperInvariant()) {
        "ARM64" { return "arm64" }
        "AMD64" { return "x64" }
        default { throw "Unsupported Windows architecture: $native (need x64 or ARM64)" }
    }
}

function Get-AppVersion {
    $text = Get-Content (Join-Path $Root "pyproject.toml") -Raw
    if ($text -match '(?m)^version\s*=\s*"([^"]+)"') {
        return $Matches[1]
    }
    return "0.2.0"
}

function Find-ISCC {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) { return $path }
    }
    $fromPath = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    return $null
}

function Install-InnoSetup {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host "Installing Inno Setup with winget..."
        & $winget.Source install --id JRSoftware.InnoSetup -e --accept-package-agreements --accept-source-agreements
        return
    }
    $installer = Join-Path $env:TEMP "innosetup-install.exe"
    Write-Host "Downloading Inno Setup..."
    Invoke-WebRequest -Uri "https://jrsoftware.org/download.php/is.exe" -OutFile $installer
    Start-Process -FilePath $installer -ArgumentList "/VERYSILENT", "/NORESTART", "/ALLUSERS" -Wait
}

function Invoke-Native {
    param([Parameter(Mandatory = $true)][string]$FilePath, [string[]]$ArgumentList = @())
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
    }
}

if ($Universal) { $Arch = "universal" }

$hostArch = Get-NativeArch
if ($Arch -eq "auto") { $Arch = $hostArch }

$Version = Get-AppVersion
Write-Host "=== Inferna Windows package ===" -ForegroundColor Cyan
Write-Host "Host: $hostArch   Target: $Arch   Version: $Version"

if ($Arch -eq "universal") {
    foreach ($needed in @("x64", "arm64")) {
        $folder = Join-Path $Root "dist\Inferna-$needed"
        if (-not (Test-Path (Join-Path $folder "Inferna.exe"))) {
            throw "Universal installer needs dist\Inferna-$needed\Inferna.exe. Build that arch on a matching PC first."
        }
    }
} else {
    if ($Arch -ne $hostArch) {
        throw "Cannot cross-compile $Arch on a $hostArch host. Build $Arch on a $Arch PC, then use -Universal."
    }

    $python = (Get-Command python -ErrorAction Stop).Source
    Write-Host "Python: $python"
    $pyVersion = & $python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
    if ([version]$pyVersion -ge [version]"3.13") {
        Write-Host "Warning: packaging is validated on Python 3.10-3.12. This is $pyVersion." -ForegroundColor Yellow
    }

    if (-not (Test-Path ".venv")) {
        Write-Host "Creating virtual environment..."
        Invoke-Native $python @("-m", "venv", ".venv")
    }

    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    $venvPip = Join-Path $Root ".venv\Scripts\pip.exe"

    Write-Host "Installing packaging dependencies..."
    Invoke-Native $venvPip @("install", "--upgrade", "pip", "wheel")

    if (-not $SkipTorch) {
        if ($Arch -eq "x64") {
            Write-Host "Installing CUDA PyTorch (cu128) for x64..."
            & $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
            if ($LASTEXITCODE -ne 0) {
                Write-Host "CUDA wheel failed; falling back to CPU PyTorch." -ForegroundColor Yellow
                Invoke-Native $venvPip @("install", "torch", "torchvision", "torchaudio")
            }
        } else {
            Write-Host "Installing default PyTorch for ARM64 (no CUDA wheel)..."
            Invoke-Native $venvPip @("install", "torch", "torchvision", "torchaudio")
        }
    }

    if (Test-Path (Join-Path $Root "requirements.txt")) {
        Invoke-Native $venvPip @("install", "-r", (Join-Path $Root "requirements.txt"))
    }
    # Quote extras so PowerShell does not treat [dev,analysis] as a wildcard.
    Invoke-Native $venvPip @("install", "-e", ".[dev,analysis]")

    Write-Host "Verifying Hugging Face serve backend..."
    Invoke-Native $venvPython @(
        "-c",
        "import torch, transformers, tokenizers, safetensors, huggingface_hub"
    )

    if (-not $SkipTests) {
        Write-Host "Running unit tests and static checks..."
        Invoke-Native $venvPython @("-m", "pytest", "-q")
        Invoke-Native $venvPython @("-m", "ruff", "check", "finetuner", "tests")
    }

    $env:FINETUNER_ARCH = $Arch
    Write-Host "Running PyInstaller ($Arch)..."
    Invoke-Native $venvPython @("-m", "PyInstaller", "finetuner.spec", "--noconfirm")

    $collected = Join-Path $Root "dist\Inferna"
    $distDir = Join-Path $Root "dist\Inferna-$Arch"
    if (-not (Test-Path (Join-Path $collected "Inferna.exe"))) {
        throw "Build failed: dist\Inferna\Inferna.exe not found"
    }
    foreach ($bundled in @("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub")) {
        $bundledDir = Join-Path $collected "_internal\$bundled"
        if (-not (Test-Path $bundledDir)) {
            throw "Installer payload is missing the Hugging Face serve backend: $bundledDir"
        }
    }
    if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
    Move-Item $collected $distDir

    $zipPath = Join-Path $Root "dist\Inferna-windows-$Arch.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $distDir "*") -DestinationPath $zipPath -Force
    Write-Host "Portable zip: $zipPath" -ForegroundColor Green
}

if ($SkipInstaller) {
    Write-Host "Skipping Inno Setup (--SkipInstaller)."
    exit 0
}

$iscc = Find-ISCC
if (-not $iscc -and $InstallInno) {
    Install-InnoSetup
    $iscc = Find-ISCC
}

if (-not $iscc) {
    Write-Host "Inno Setup 6.3+ not found. Install from https://jrsoftware.org/isinfo.php or rerun with -InstallInno." -ForegroundColor Yellow
    if ($Arch -ne "universal") {
        Write-Host "Portable zip is ready at dist\Inferna-windows-$Arch.zip"
    }
    exit 0
}

$issPath = Join-Path $Root "scripts\finetuner_installer.iss"
$distDefine = "..\dist\Inferna-$Arch"
Write-Host "Compiling installer ($Arch) with $iscc"
Invoke-Native $iscc @(
    "/DAppArch=$Arch",
    "/DAppVersion=$Version",
    "/DDistFolder=$distDefine",
    $issPath
)

$setup = Join-Path $Root "dist\Inferna-Setup-$Arch.exe"
if (-not (Test-Path $setup)) {
    throw "ISCC finished but $setup was not created"
}
Write-Host "Installer: $setup" -ForegroundColor Green
