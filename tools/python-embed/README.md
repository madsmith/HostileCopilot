# Python embeddable runtime (dev-only)

This folder is intended to hold the **Windows embeddable Python distribution** (the `*-embed-amd64.zip` from python.org) during development and bundle assembly.

The embedded runtime is **not committed** to git. See `.gitignore` in this directory.

## Download

1) Open:

`https://www.python.org/downloads/windows/`

2) Locate the Python version you want to ship (this project targets Python 3.13).

3) Under **Files**, download:

- `Windows embeddable package (64-bit)`

This downloads a zip such as:

- `python-3.12.x-embed-amd64.zip`

## Install (extract)

Extract the contents of the zip **into this folder** (`tools/launcher/python-embed/`).

After extraction, this folder should contain at least:

- `python.exe`
- `pythonw.exe`
- `python313.dll`

## Using with the bundle assembler

When assembling a distributable bundle, pass this directory as `-EmbeddedPythonDir`:

```powershell
powershell -ExecutionPolicy Bypass -File tools/launcher/assemble_bundle_and_zip.ps1 `
  -EmbeddedPythonDir "tools\launcher\python-embed" `
  -RebuildVeneers -RebuildWheel
```

## Git ignore

This directory contains a `.gitignore` that ignores **all extracted files** and keeps only:

- `README.md`
- `.gitignore`
