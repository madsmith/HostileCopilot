param(
  [string]$OutDir = "dist-launcher"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
  $Py = $VenvPython
} else {
  $Py = "python"
}

& $Py -m pip install --upgrade pyinstaller

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Push-Location $PSScriptRoot
try {
  & $Py -m PyInstaller --noconfirm --clean --distpath (Join-Path $RepoRoot $OutDir) HostileActiveScanner.spec
  & $Py -m PyInstaller --noconfirm --clean --distpath (Join-Path $RepoRoot $OutDir) HostileCoPilot.spec
} finally {
  Pop-Location
}
