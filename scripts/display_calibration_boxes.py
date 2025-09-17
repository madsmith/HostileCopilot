import sys
import asyncio
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hostile_copilot.config import load_config, OmegaConfig
from hostile_copilot.ui.overlay import Overlay


def get_app_config() -> OmegaConfig:
    """
    Load the main config (config.yaml + _private) and then load the app config
    (settings.yaml by default) using the same loader to include local overrides.
    """
    root_cfg = load_config()  # defaults to config/config.yaml + _private
    app_cfg_path = root_cfg.get("app.config", "config/settings.yaml")
    app_cfg_path = Path(app_cfg_path)
    # If relative, resolve from project root
    if not app_cfg_path.is_absolute():
        app_cfg_path = PROJECT_ROOT / app_cfg_path

    # Ensure the file exists to mirror app behavior (HostileCoPilotApp.__init__ touches it)
    if not app_cfg_path.exists():
        app_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        app_cfg_path.touch()

    app_cfg = load_config(app_cfg_path)
    return app_cfg


def _try_get(cfg: OmegaConfig, key: str, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def build_rectangles(app_cfg: OmegaConfig):
    """
    Build a list of rectangles to draw from the calibration section.
    Each rectangle is (x, y, w, h).
    - mining_scan: bounding box from start_x/start_y to end_x/end_y
    - ping_scan: bounding box from start_x/start_y to end_x/end_y
    - nav_search.location: small square centered on the point
    - nav_locations.(search|first_result|route): small squares centered on the points
    """
    rects: list[tuple[int, int, int, int]] = []

    # Helper to add a centered square around a point
    def add_point_square(x_key: str, y_key: str, size: int = 9):
        x = _try_get(app_cfg, x_key)
        y = _try_get(app_cfg, y_key)
        if x is None or y is None:
            return
        half = size // 2
        rects.append((int(x) - half, int(y) - half, size, size))

    # mining_scan bbox
    ms_sx = _try_get(app_cfg, "calibration.mining_scan.start_x")
    ms_sy = _try_get(app_cfg, "calibration.mining_scan.start_y")
    ms_ex = _try_get(app_cfg, "calibration.mining_scan.end_x")
    ms_ey = _try_get(app_cfg, "calibration.mining_scan.end_y")
    if all(v is not None for v in [ms_sx, ms_sy, ms_ex, ms_ey]):
        x0, y0, x1, y1 = int(ms_sx), int(ms_sy), int(ms_ex), int(ms_ey)
        rects.append((min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))

    # ping_scan bbox
    ps_sx = _try_get(app_cfg, "calibration.ping_scan.start_x")
    ps_sy = _try_get(app_cfg, "calibration.ping_scan.start_y")
    ps_ex = _try_get(app_cfg, "calibration.ping_scan.end_x")
    ps_ey = _try_get(app_cfg, "calibration.ping_scan.end_y")
    if all(v is not None for v in [ps_sx, ps_sy, ps_ex, ps_ey]):
        x0, y0, x1, y1 = int(ps_sx), int(ps_sy), int(ps_ex), int(ps_ey)
        rects.append((min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))

    # nav_search.location point
    add_point_square("calibration.nav_search.location.x", "calibration.nav_search.location.y")

    # nav_locations points
    add_point_square("calibration.nav_locations.search.x", "calibration.nav_locations.search.y")
    add_point_square("calibration.nav_locations.first_result.x", "calibration.nav_locations.first_result.y")
    add_point_square("calibration.nav_locations.route.x", "calibration.nav_locations.route.y")

    return rects

async def main():
    app_cfg = get_app_config()
    rects = build_rectangles(app_cfg)

    if not rects:
        print("No calibration rectangles found in settings. Exiting.")
        return

    print("Rendering calibration rectangles:")
    for r in rects:
        print(f" - x={r[0]} y={r[1]} w={r[2]} h={r[3]}")

    overlay = Overlay()
    for (x, y, w, h) in rects:
        overlay.add_rect(x, y, w, h)

    print("Press Ctrl+C in this console to exit the overlay.")
    try:
        await overlay.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    asyncio.run(main())
