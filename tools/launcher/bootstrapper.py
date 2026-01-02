from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BundleLayout:
    root: Path

    @property
    def python_exe(self) -> Path:
        return self.root / "python" / "python.exe"

    @property
    def payload_dir(self) -> Path:
        return self.root / "payload"

    @property
    def app_dir(self) -> Path:
        return self.root / "app"

    @property
    def venv_dir(self) -> Path:
        return self.app_dir / "venv"

    @property
    def venv_python_exe(self) -> Path:
        return self.venv_dir / "Scripts" / "python.exe"

    @property
    def venv_scripts_dir(self) -> Path:
        return self.venv_dir / "Scripts"


def _bundle_root() -> Path:
    return Path(sys.argv[0]).resolve().parent


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd else None)


def _ensure_dirs(layout: BundleLayout) -> None:
    layout.app_dir.mkdir(parents=True, exist_ok=True)


def _enable_embedded_site_packages(layout: BundleLayout) -> None:
    # Windows embeddable Python uses a python*._pth file to isolate sys.path.
    # If present, we must enable `import site` and include Lib\site-packages,
    # otherwise pip installed via get-pip.py will not be importable.
    pth_files = sorted((layout.python_exe.parent).glob("python*._pth"))
    if not pth_files:
        return

    pth_path = pth_files[0]
    text = pth_path.read_text(encoding="utf-8")
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]

    want_site_packages = r"Lib\\site-packages"
    if not any(ln.strip() == want_site_packages for ln in lines):
        # Place it near the end but before import site.
        insert_at = len(lines)
        for i, ln in enumerate(lines):
            if ln.strip().startswith("import site") or ln.strip().startswith("#import site"):
                insert_at = i
                break
        lines.insert(insert_at, want_site_packages)

    # Ensure site is imported.
    found_site_line = False
    for i, ln in enumerate(lines):
        if ln.strip() == "import site":
            found_site_line = True
            break
        if ln.strip() == "#import site":
            lines[i] = "import site"
            found_site_line = True
            break

    if not found_site_line:
        lines.append("import site")

    new_text = "\n".join(lines) + "\n"
    if new_text != text:
        pth_path.write_text(new_text, encoding="utf-8")


def _create_venv(layout: BundleLayout) -> None:
    if layout.venv_python_exe.exists():
        return

    if not layout.python_exe.exists():
        raise FileNotFoundError(
            f"Embedded Python not found at: {layout.python_exe}. "
            "Expected bundle layout: <root>/python/python.exe"
        )

    _enable_embedded_site_packages(layout)

    try:
        _run([str(layout.python_exe), "-m", "venv", str(layout.venv_dir)])
        return
    except subprocess.CalledProcessError:
        # Python embeddable distribution often does not include stdlib venv.
        pass

    _create_venv_via_virtualenv(layout)


def _ensure_embedded_pip(layout: BundleLayout) -> None:
    _enable_embedded_site_packages(layout)
    try:
        _run([str(layout.python_exe), "-m", "pip", "--version"])
        return
    except subprocess.CalledProcessError:
        pass

    # Try stdlib ensurepip first.
    try:
        _run([str(layout.python_exe), "-m", "ensurepip", "--upgrade"])
        return
    except subprocess.CalledProcessError:
        pass

    get_pip = layout.payload_dir / "get-pip.py"
    if not get_pip.exists():
        raise FileNotFoundError(
            f"Embedded Python has no pip/ensurepip and payload is missing: {get_pip}. "
            "Add get-pip.py to the bundle payload."
        )

    _run([str(layout.python_exe), str(get_pip)])

    # After get-pip, ensure path isolation isn't preventing imports.
    _enable_embedded_site_packages(layout)


def _create_venv_via_virtualenv(layout: BundleLayout) -> None:
    _ensure_embedded_pip(layout)

    _run([str(layout.python_exe), "-m", "pip", "install", "--upgrade", "virtualenv"])
    _run([str(layout.python_exe), "-m", "virtualenv", str(layout.venv_dir)])


def _ensure_pip(layout: BundleLayout) -> None:
    try:
        _run([str(layout.venv_python_exe), "-m", "pip", "--version"])
        return
    except subprocess.CalledProcessError:
        pass

    # Try to bootstrap pip using embedded python.
    _ensure_embedded_pip(layout)


def _upgrade_pip(layout: BundleLayout) -> None:
    _run([str(layout.venv_python_exe), "-m", "pip", "install", "--upgrade", "pip"])


def _find_payload_wheel(layout: BundleLayout) -> Path:
    wheels = sorted(layout.payload_dir.glob("HostileCoPilot-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            f"No HostileCoPilot wheel found in: {layout.payload_dir}. "
            "Expected a file like HostileCoPilot-<version>-py3-none-any.whl"
        )
    return wheels[-1]


def _install_wheel(layout: BundleLayout) -> None:
    wheel = _find_payload_wheel(layout)
    _run([str(layout.venv_python_exe), "-m", "pip", "install", "--upgrade", str(wheel)])


def _run_cuda_install(layout: BundleLayout) -> None:
    cuda_install = layout.payload_dir / "cuda_install.py"
    if not cuda_install.exists():
        raise FileNotFoundError(
            f"cuda_install.py not found in payload: {cuda_install}. "
            "Copy the repo-root cuda_install.py into <bundle>/payload/cuda_install.py"
        )

    _run([str(layout.venv_python_exe), str(cuda_install)])


def _sync_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    dst.mkdir(parents=True, exist_ok=True)

    # Copy files recursively; overwrite to keep payload authoritative.
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (target_root / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(Path(root) / f, target_root / f)


def _sync_assets(layout: BundleLayout) -> None:
    _sync_tree(layout.payload_dir / "config", layout.app_dir / "config")
    _sync_tree(layout.payload_dir / "resources", layout.app_dir / "resources")


def _run_console_script(layout: BundleLayout, script_name: str, args: Sequence[str]) -> int:
    exe = layout.venv_scripts_dir / (script_name + ".exe")
    if not exe.exists():
        raise FileNotFoundError(
            f"Expected console script not found: {exe}. "
            "The package install may have failed, or the script name changed."
        )

    completed = subprocess.run([str(exe), *args], cwd=str(layout.app_dir))
    return int(completed.returncode)


def bootstrap_and_run(*, target: str, args: Sequence[str]) -> int:
    layout = BundleLayout(root=_bundle_root())

    _ensure_dirs(layout)
    _create_venv(layout)
    _ensure_pip(layout)
    _upgrade_pip(layout)

    _install_wheel(layout)
    _run_cuda_install(layout)
    _sync_assets(layout)

    if target == "active_scanner":
        return _run_console_script(layout, "hostile-copilot-active-scanner", args)

    if target == "hostile_copilot":
        return _run_console_script(layout, "hostile-copilot", args)

    raise ValueError(f"Unknown target: {target}")
