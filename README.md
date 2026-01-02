# HostileCoPilot

> Your comprehensive Star Citizen companion application

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

## Overview

HostileCoPilot is a powerful companion application designed to enhance your Star Citizen gaming experience. Whether you're a seasoned pilot or just starting your journey in the 'verse, this tool provides essential features to help you navigate, trade, and thrive in the Star Citizen universe.

## Features

- 🚀 **Quick Launch**: Easy-to-use command-line interface
- 📊 **Status Monitoring**: Real-time system status checks
- ⚙️ **Configuration Management**: Flexible configuration system
- 🎮 **Star Citizen Integration**: Seamless companion functionality

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Install from Source

```bat
:: Clone the repository
git clone https://github.com/martin/HostileCoPilot.git
cd HostileCoPilot

:: Create and activate a virtual environment (cmd.exe)
py -3.12 -m venv .venv
.venv\Scripts\activate.bat

:: Install in development mode
pip install -e .

:: Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bat
:: Start the application (after activating your venv)
hostile-copilot
```

### Configuration

Credentials are required for OpenAI and UEX Corp APIs.

1. Copy the sample private config file:

```bat
copy config\config_private.sample.yaml config\config_private.yaml
```

2. Edit `config/config_private.yaml` and set your credentials:

```yaml
private:
  openai:
    api_key: "your-openai-api-key"
  uexcorp:
    bearer_token: "your-uexcorp-bearer-token"
```

3. The public config at `config/config.yaml` references these values via `${private...}` interpolation, so you generally don't need to edit it, but any similar keys defined in `config/config_private.yaml` will override the public config.

```yaml
openai:
  api_key: "${private.openai.api_key}"
  api_base: "https://api.openai.com/v1"
uexcorp:
  bearer_token: "${private.uexcorp.bearer_token}"
```

## Development

### Setting up Development Environment

```bat
:: Clone the repository
git clone https://github.com/martin/HostileCoPilot.git
cd HostileCoPilot

:: Create virtual environment
uv venv
.venv\Scripts\activate.bat

:: Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## Building Standalone Executable

The standalone build is a single-folder bundle containing:

- Wrapper executables (`HostileActiveScanner.exe`, `HostileCoPilot.exe`)
- An embedded Python distribution
- A `payload/` folder with the HostileCoPilot wheel, `cuda_install.py`, and required assets

On first run, the wrapper bootstraps a venv under `./app/`, installs the wheel from `./payload/`, runs `payload/cuda_install.py` to install a suitable CUDA-enabled PyTorch build, then launches the selected entry point.

### 1) Download Windows embeddable Python (3.12)

- Download the **Windows embeddable package** for Python **3.12.x** from:
  - https://www.python.org/downloads/windows/
- Extract it to:
  - `tools/python-embed/<python embed version>` (e.g., `python-3.12.10-embed-amd64`)

That folder should contain `python.exe`.

### 2) Build wrappers + assemble the bundle

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\launcher\assemble_bundle_and_zip.ps1 `
  -EmbeddedPythonDir "tools\python-embed\python-3.12.10-embed-amd64" `
  -RebuildWrappers `
  -RebuildWheel
```

Outputs:

- `dist-wrapper/` (wrapper executables)
- `dist-bundle/HostileCoPilotBundle/` (final folder)
- `dist-bundle/HostileCoPilotBundle.zip`

### 3) Run

Run either executable from the bundle folder:

- `dist-bundle/HostileCoPilotBundle/HostileActiveScanner.exe`
- `dist-bundle/HostileCoPilotBundle/HostileCoPilot.exe`

## Support

- 📖 [Documentation](https://github.com/martin/HostileCoPilot/wiki)
- 🐛 [Issue Tracker](https://github.com/martin/HostileCoPilot/issues)
- 💬 [Discussions](https://github.com/martin/HostileCoPilot/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Third-Party Notices

Portions of the virtual keyboard functionality in `src/hostile_copilot/utils/input/gremlin/` are adapted from the Joystick Gremlin project:

- Joystick Gremlin — https://github.com/WhiteMagic/JoystickGremlin
- Copyright © 2015–2024 Lionel Ott
- Licensed under the GNU General Public License v3.0 (GPLv3)

Those adapted parts are used under the terms of the GPLv3 license. Please refer to the Joystick Gremlin repository for the full license text and additional details. Redistribution of builds including these portions may be subject to GPLv3 requirements.

## Acknowledgments

- Star Citizen community for inspiration and feedback
- Roberts Space Industries for creating the Star Citizen universe
- All contributors who help make this project better

---

**Disclaimer**: This is an unofficial companion application and is not affiliated with or endorsed by Cloud Imperium Games or Roberts Space Industries.
