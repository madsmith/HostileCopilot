param(
  [string]$BundleName = "HostileCoPilotBundle",
  [string]$OutRoot = "dist-bundle",
  [string]$VeneerOutDir = "dist-launcher",
  [string]$EmbeddedPythonDir = "",
  [switch]$RebuildVeneers,
  [switch]$RebuildWheel
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

if ($RebuildVeneers) {
  & "$PSScriptRoot\build_veneer_exes.ps1" -OutDir $VeneerOutDir
}

$BundleRoot = Join-Path $RepoRoot $OutRoot
$BundleDir = Join-Path $BundleRoot $BundleName

# Clean bundle dir
if (Test-Path $BundleDir) {
  Remove-Item -Recurse -Force $BundleDir
}
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

# Copy veneer exes
$VeneerDir = Join-Path $RepoRoot $VeneerOutDir
Copy-Item -Force (Join-Path $VeneerDir "HostileActiveScanner.exe") (Join-Path $BundleDir "HostileActiveScanner.exe")
Copy-Item -Force (Join-Path $VeneerDir "HostileCoPilot.exe") (Join-Path $BundleDir "HostileCoPilot.exe")

# Embedded python
if (-not $EmbeddedPythonDir) {
  throw "EmbeddedPythonDir is required. Point it at an extracted Python embeddable folder containing python.exe. Download: https://www.python.org/downloads/windows/"
}
if (-not (Test-Path (Join-Path $EmbeddedPythonDir "python.exe"))) {
  throw "EmbeddedPythonDir does not look like an embedded Python folder (python.exe missing): $EmbeddedPythonDir"
}
Copy-Item -Recurse -Force $EmbeddedPythonDir (Join-Path $BundleDir "python")

# Payload
$PayloadDir = Join-Path $BundleDir "payload"
New-Item -ItemType Directory -Force -Path $PayloadDir | Out-Null

# Ensure get-pip.py is present for embedded python bootstrapping (venv/ensurepip are often missing)
$GetPipPath = Join-Path $PayloadDir "get-pip.py"
if (-not (Test-Path $GetPipPath)) {
  Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPipPath
}

# Build wheel
if ($RebuildWheel) {
  & "$RepoRoot\.venv\Scripts\python.exe" -m pip install --upgrade build
  & "$RepoRoot\.venv\Scripts\python.exe" -m build --wheel
}

$DistDir = Join-Path $RepoRoot "dist"
if (-not (Test-Path $DistDir)) {
  throw "Expected wheel output under $DistDir (run python -m build --wheel)"
}
$Wheel = Get-ChildItem -Path $DistDir -Filter "HostileCoPilot-*.whl" | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $Wheel) {
  throw "No HostileCoPilot wheel found under $DistDir"
}
Copy-Item -Force $Wheel.FullName (Join-Path $PayloadDir $Wheel.Name)

# Copy cuda_install.py
Copy-Item -Force (Join-Path $RepoRoot "cuda_install.py") (Join-Path $PayloadDir "cuda_install.py")

# Copy required runtime assets
New-Item -ItemType Directory -Force -Path (Join-Path $PayloadDir "config") | Out-Null
Copy-Item -Force (Join-Path $RepoRoot "config\active_scanner.yaml") (Join-Path $PayloadDir "config\active_scanner.yaml")
Copy-Item -Recurse -Force (Join-Path $RepoRoot "resources") (Join-Path $PayloadDir "resources")

# Create empty app dir (optional)
New-Item -ItemType Directory -Force -Path (Join-Path $BundleDir "app") | Out-Null

# Zip
New-Item -ItemType Directory -Force -Path $BundleRoot | Out-Null
$ZipPath = Join-Path $BundleRoot ("$BundleName.zip")
if (Test-Path $ZipPath) {
  Remove-Item -Force $ZipPath
}
Compress-Archive -Path $BundleDir -DestinationPath $ZipPath

Write-Host "Bundle created: $BundleDir"
Write-Host "Zip created: $ZipPath"
