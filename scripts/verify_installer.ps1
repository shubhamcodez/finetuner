# Compile the Inno scripts against stub payloads (no PyInstaller).
param(
    [switch]$InstallInno
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

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

if ($InstallInno -and -not (Find-ISCC)) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        & $winget.Source install --id JRSoftware.InnoSetup -e --accept-package-agreements --accept-source-agreements
    } else {
        $installer = Join-Path $env:TEMP "innosetup-install.exe"
        Invoke-WebRequest -Uri "https://jrsoftware.org/download.php/is.exe" -OutFile $installer
        Start-Process -FilePath $installer -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
    }
}

$iscc = Find-ISCC
if (-not $iscc) {
    throw "Inno Setup 6.3+ is required to verify the installer. Rerun with -InstallInno."
}

foreach ($arch in @("x64", "arm64")) {
    $dir = Join-Path $Root "dist\Finetuner-$arch"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Set-Content -Path (Join-Path $dir "Finetuner.exe") -Value "stub"
}

$iss = Join-Path $Root "scripts\finetuner_installer.iss"
foreach ($arch in @("x64", "arm64", "universal")) {
    Write-Host "Compiling Finetuner-Setup-$arch.exe"
    & $iscc "/DAppArch=$arch" "/DAppVersion=0.2.0" "/DDistFolder=..\dist\Finetuner-$arch" $iss
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed for $arch" }
    $setup = Join-Path $Root "dist\Finetuner-Setup-$arch.exe"
    if (-not (Test-Path $setup)) { throw "Missing $setup" }
}

Write-Host "Installer scripts compile for x64, ARM64, and universal." -ForegroundColor Green
