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


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd else None, env=merged_env)


def _print_step(text: str) -> None:
    print(text, flush=True)


def _cmd_succeeds_quiet(cmd: Sequence[str]) -> bool:
    try:
        subprocess.run(list(cmd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _ensure_dirs(layout: BundleLayout) -> bool:
    if layout.app_dir.exists():
        return False
    layout.app_dir.mkdir(parents=True, exist_ok=True)
    return True


def _venv_has_pip(layout: BundleLayout) -> bool:
    return _cmd_succeeds_quiet([str(layout.venv_python_exe), "-m", "pip", "--version"])


def _embedded_has_pip(layout: BundleLayout) -> bool:
    _enable_embedded_site_packages(layout)
    return _cmd_succeeds_quiet([str(layout.python_exe), "-m", "pip", "--version"])


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


def _create_venv(layout: BundleLayout) -> bool:
    if layout.venv_python_exe.exists():
        return False

    if not layout.python_exe.exists():
        raise FileNotFoundError(
            f"Embedded Python not found at: {layout.python_exe}. "
            "Expected bundle layout: <root>/python/python.exe"
        )

    _enable_embedded_site_packages(layout)

    try:
        _run([str(layout.python_exe), "-m", "venv", str(layout.venv_dir)])
        return True
    except subprocess.CalledProcessError:
        # Python embeddable distribution often does not include stdlib venv.
        pass

    _create_venv_via_virtualenv(layout)
    return True


def _ensure_embedded_pip(layout: BundleLayout) -> bool:
    if _embedded_has_pip(layout):
        return False

    # Try stdlib ensurepip first.
    try:
        _run([str(layout.python_exe), "-m", "ensurepip", "--upgrade"])
        return True
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
    return True


def _create_venv_via_virtualenv(layout: BundleLayout) -> None:
    _ensure_embedded_pip(layout)

    _run([str(layout.python_exe), "-m", "pip", "install", "--upgrade", "virtualenv"])
    _run([str(layout.python_exe), "-m", "virtualenv", str(layout.venv_dir)])


def _ensure_pip(layout: BundleLayout) -> bool:
    if _venv_has_pip(layout):
        return False

    # Try to bootstrap pip using embedded python.
    return _ensure_embedded_pip(layout)


def _upgrade_pip(layout: BundleLayout) -> bool:
    marker = layout.app_dir / ".pip_upgraded"
    if marker.exists():
        return False
    _run([str(layout.venv_python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    marker.write_text("ok\n", encoding="utf-8")
    return True


def _uv_exe(layout: BundleLayout) -> Path | None:
    for name in ("uv.exe", "uv.cmd"):
        p = layout.venv_scripts_dir / name
        if p.exists():
            return p
    return None


def _needs_uv(layout: BundleLayout) -> bool:
    return _uv_exe(layout) is None


def _ensure_uv(layout: BundleLayout) -> bool:
    if not _needs_uv(layout):
        return False

    _run([str(layout.venv_python_exe), "-m", "pip", "install", "--upgrade", "uv"])
    return True


def _uv_env(layout: BundleLayout) -> dict[str, str]:
    return {"UV_CACHE_DIR": str(layout.app_dir / ".uv-cache")}


def _find_payload_wheel(layout: BundleLayout) -> Path:
    wheels = sorted(layout.payload_dir.glob("HostileCoPilot-*.whl"))
    if not wheels:
        raise FileNotFoundError(
            f"No HostileCoPilot wheel found in: {layout.payload_dir}. "
            "Expected a file like HostileCoPilot-<version>-py3-none-any.whl"
        )
    return wheels[-1]


def _wheel_version_from_filename(wheel: Path) -> str | None:
    name = wheel.name
    if not name.startswith("HostileCoPilot-"):
        return None
    parts = name.split("-")
    if len(parts) < 2:
        return None
    return parts[1]


def _installed_hostile_copilot_version(layout: BundleLayout) -> str | None:
    try:
        completed = subprocess.run(
            [str(layout.venv_python_exe), "-m", "pip", "show", "HostileCoPilot"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    for line in completed.stdout.splitlines():
        if line.lower().startswith("version:"):
            return line.split(":", 1)[1].strip()
    return None


def _needs_wheel_install(layout: BundleLayout) -> bool:
    marker = layout.app_dir / ".installed_wheel"
    wheel = _find_payload_wheel(layout)
    if marker.exists():
        installed = marker.read_text(encoding="utf-8").strip()
        if installed == wheel.name:
            return False
    want_version = _wheel_version_from_filename(wheel)
    if not want_version:
        return True
    have_version = _installed_hostile_copilot_version(layout)
    return have_version != want_version


def _install_wheel(layout: BundleLayout) -> bool:
    wheel = _find_payload_wheel(layout)
    want_version = _wheel_version_from_filename(wheel)
    have_version = _installed_hostile_copilot_version(layout) if want_version else None
    if want_version and have_version == want_version:
        return False

    uv = _uv_exe(layout)
    if uv is not None:
        _run(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(layout.venv_python_exe),
                "--upgrade",
                str(wheel),
            ],
            env=_uv_env(layout),
        )
        (layout.app_dir / ".installed_wheel").write_text(wheel.name + "\n", encoding="utf-8")
        return True

    _run([str(layout.venv_python_exe), "-m", "pip", "install", "--upgrade", str(wheel)])
    (layout.app_dir / ".installed_wheel").write_text(wheel.name + "\n", encoding="utf-8")
    return True


def _run_cuda_install(layout: BundleLayout) -> bool:
    marker = layout.app_dir / ".cuda_installed"
    if marker.exists():
        return False

    cuda_install = layout.payload_dir / "cuda_install.py"
    if not cuda_install.exists():
        raise FileNotFoundError(
            f"cuda_install.py not found in payload: {cuda_install}. "
            "Copy the repo-root cuda_install.py into <bundle>/payload/cuda_install.py"
        )

    _run([str(layout.venv_python_exe), str(cuda_install)])
    marker.write_text("ok\n", encoding="utf-8")
    return True


def _needs_cuda_install(layout: BundleLayout) -> bool:
    return not (layout.app_dir / ".cuda_installed").exists()


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


def _sync_assets(layout: BundleLayout) -> bool:
    config_dst = layout.app_dir / "config"
    resources_dst = layout.app_dir / "resources"

    needs = False
    if (layout.payload_dir / "config").exists() and not config_dst.exists():
        needs = True
    if (layout.payload_dir / "resources").exists() and not resources_dst.exists():
        needs = True
    if not needs:
        return False

    _sync_tree(layout.payload_dir / "config", config_dst)
    _sync_tree(layout.payload_dir / "resources", resources_dst)
    return True


def _needs_asset_sync(layout: BundleLayout) -> bool:
    config_dst = layout.app_dir / "config"
    resources_dst = layout.app_dir / "resources"
    if (layout.payload_dir / "config").exists() and not config_dst.exists():
        return True
    if (layout.payload_dir / "resources").exists() and not resources_dst.exists():
        return True
    return False


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

    if not layout.app_dir.exists():
        _ensure_dirs(layout)

    created_venv = False
    if not layout.venv_python_exe.exists():
        _print_step("[Bootstrap] Creating virtual environment")
        created_venv = _create_venv(layout)

    ensured_pip = False
    if not _venv_has_pip(layout):
        _print_step("[Bootstrap] Bootstrapping pip")
        ensured_pip = _ensure_pip(layout)

    # Only upgrade pip when we've just created the venv or bootstrapped pip.
    if (created_venv or ensured_pip) and not (layout.app_dir / ".pip_upgraded").exists():
        _print_step("[Bootstrap] Upgrading pip")
        _upgrade_pip(layout)

    if _needs_uv(layout):
        _print_step("[Bootstrap] Installing uv")
        _ensure_uv(layout)

    if _needs_wheel_install(layout):
        _print_step("[Bootstrap] Installing HostileCoPilot")
        _install_wheel(layout)

    if _needs_cuda_install(layout):
        _print_step("[Bootstrap] Installing CUDA-enabled PyTorch")
        _run_cuda_install(layout)

    if _needs_asset_sync(layout):
        _print_step("[Bootstrap] Syncing assets")
        _sync_assets(layout)

    if target == "active_scanner":
        return _run_console_script(layout, "hostile-copilot-active-scanner", args)

    if target == "hostile_copilot":
        return _run_console_script(layout, "hostile-copilot", args)

    raise ValueError(f"Unknown target: {target}")
