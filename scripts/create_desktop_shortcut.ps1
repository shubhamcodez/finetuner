# Creates a Desktop shortcut for Finetuner with the logo icon.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Icon = Join-Path $Root "assets\icon.ico"
$Pythonw = Join-Path $Root ".venv\Scripts\pythonw.exe"

if (-not (Test-Path $Pythonw)) {
    Write-Error "Python venv not found. Run: python -m venv .venv && .\scripts\install_gpu.ps1"
}
if (-not (Test-Path $Icon)) {
    Write-Error "Icon not found at $Icon"
}

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Finetuner.lnk"

$Wsh = New-Object -ComObject WScript.Shell
$Shortcut = $Wsh.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Pythonw
$Shortcut.Arguments = "-m finetuner"
$Shortcut.WorkingDirectory = $Root
$Shortcut.IconLocation = "$Icon,0"
$Shortcut.Description = "Finetuner GPU LLM Fine-tuning and Evaluation"
$Shortcut.Save()

Write-Host "Desktop shortcut created: $ShortcutPath" -ForegroundColor Green
