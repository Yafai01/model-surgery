# Model Surgery setup for Windows 11: creates .venv, installs the backend, builds the frontend.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)"
    py -3.11 -m venv .venv
    if (-not $?) { python -m venv .venv }
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e "backend[dev]"

Push-Location frontend
if (Test-Path "package-lock.json") { npm ci } else { npm install }
npm run build
Pop-Location

Write-Host ""
Write-Host "Done. Start with:  .\.venv\Scripts\model-surgery serve --open"
Write-Host "Optional:          --llama-cpp-dir <folder containing llama-quantize.exe>"
