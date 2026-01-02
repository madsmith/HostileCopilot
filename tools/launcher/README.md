# Launcher (PyInstaller)

This folder contains a small bootstrapper that creates/uses a venv under `./app/`, installs HostileCoPilot from a wheel under `./payload/`, runs `payload/cuda_install.py` to pick an appropriate torch build, then launches the selected console script.

## Build the veneer executables

From the repo root:

- `powershell -ExecutionPolicy Bypass -File tools/launcher/build_veneer_exes.ps1`

Outputs are placed under `dist-launcher/`.

## Assemble a distributable bundle + zip

This creates a single folder with the expected runtime layout and then zips it.

You must download/extract the Python embeddable distribution separately and point `-EmbeddedPythonDir` at that folder.

Note: the Windows embeddable Python build often does not ship with `venv` (and sometimes lacks `ensurepip`). The launcher falls back to bootstrapping `pip` using `payload/get-pip.py` and then creating the environment with `virtualenv`.

Example:

- `powershell -ExecutionPolicy Bypass -File tools/launcher/assemble_bundle_and_zip.ps1 -EmbeddedPythonDir "C:\path\to\python-embed" -RebuildVeneers -RebuildWheel`

## Expected bundle layout

After building the veneer executables, assemble a single folder for end-users:

- `HostileActiveScanner.exe`
- `HostileCoPilot.exe`
- `python/` (embedded python distribution; must contain `python.exe`)
- `payload/`
  - `HostileCoPilot-*.whl`
  - `cuda_install.py`
  - `config/active_scanner.yaml`
  - `resources/models/.../*.pt`
- `app/` (created at runtime)

## Running

- `HostileActiveScanner.exe` runs `hostile-copilot-active-scanner` inside the managed venv.
- `HostileCoPilot.exe` runs `hostile-copilot` inside the managed venv.

Both forward any command line arguments to the underlying console script.

The launcher sets the working directory to `./app` so existing relative paths like `config/active_scanner.yaml` and `resources/models/...` resolve without modifying the app code.
