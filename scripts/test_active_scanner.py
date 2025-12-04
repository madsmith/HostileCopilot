import argparse
import re
import logging
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import dxcam
from dxcam import DXFactory
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from hostile_copilot.client.components.ui.overlay import Overlay, Detection

try:
    from ultralytics import YOLO
except ImportError as e:  # type: ignore
    YOLO = None  # type: ignore

logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    if YOLO is None:
        raise RuntimeError(
            "ultralytics package is required for YOLOv11 scanning. Install with `pip install ultralytics`."
        )

    if not model_path.exists():
        model_path = model_path.with_suffix(".pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    return YOLO(str(model_path), verbose=True)


_camera: dxcam.DXCamera | None = None

def get_device_output(output: int) -> tuple[int, int]:
    factory = DXFactory()
    count = 0
    for device_idx, outputs in enumerate(factory.outputs):
        for output_idx, _ in enumerate(outputs):
            if count == output:
                return device_idx, output_idx
            count += 1

    raise IndexError(f"Index {output} is out of range.")

def grab_screenshot_bgr(output=0, region=None, debug=False) -> np.ndarray | None:
    global _camera

    if _camera is None:
        device_idx, output_idx = get_device_output(output)
        # Primary output in BGR format to match OpenCV expectations
        _camera = dxcam.create(
            device_idx=device_idx,
            output_idx=output_idx,
            output_color="BGR",
        )

    frame = _camera.grab(region)
    if frame is None:
        return None

    if debug:
        cv2.imwrite("screenshots/dxcam_test_frame.png", frame)
    return frame

def class_color(class_id: int, total_classes: int = 20, opacity=1.0) -> QColor:
    # distribute hues evenly around the color wheel
    hue = int((class_id % total_classes) * (360 / total_classes))
    sat = 255
    val = 255
    return QColor.fromHsv(hue, sat, val, int(opacity * 255))

def extract_detections(results, threshold: float) -> list[Detection]:
    if results is None:
        return []

    if not isinstance(results, (list, tuple)):
        results = [results]

    detections: list[Detection] = []

    for result in results:
        obbs = getattr(result, "obb", None)
        if obbs is None:
            continue

        for obb in obbs:
            class_id = int(obb.cls[0])
            class_name = result.names[class_id]
            confidence = float(obb.conf[0])

            if confidence < threshold:
                print(f"Skipping {class_name} with confidence {confidence}")
                continue

            corners = obb.xyxyxyxy[0].cpu().numpy().astype("float32")  # shape (4, 2)
            xs = corners[:, 0]
            ys = corners[:, 1]

            x_min = int(xs.min())
            y_min = int(ys.min())
            x_max = int(xs.max())
            y_max = int(ys.max())

            # assign a color based on class id 0-10
            color = class_color(class_id, 10, .3)

            # Sanity check to avoid degenerate or wildly invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            label = f"{class_name} {confidence:.2f}"
            detection = Detection(
                x1=x_min,
                y1=y_min,
                x2=x_max,
                y2=y_max,
                label=label,
                color=color,
            )
            detections.append(detection)

    return detections
    
def crop_rotated_region(image, corners):
    (tl, tr, br, bl) = corners

    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped


def scale_detections(
    detections: list[Detection],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> list[Detection]:
    """Scale detection coordinates from source image size to destination size.

    This accounts for differences between the screenshot resolution and the
    overlay window's logical size (e.g. due to DPI scaling).
    """

    if src_width <= 0 or src_height <= 0:
        return []

    sx = dst_width / src_width
    sy = dst_height / src_height

    scaled: list[Detection] = []
    for det in detections:
        x1 = int(det.x1 * sx)
        y1 = int(det.y1 * sy)
        x2 = int(det.x2 * sx)
        y2 = int(det.y2 * sy)

        scaled.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, label=det.label, color=det.color))

    return scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test active YOLO scanner on periodic screenshots")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="scanner.pt",
        help="Model filename under resources/models/scanner/",
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default=None,
        help="Window size for Yolo inference",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Scan interval in seconds between screenshots",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    model_path = project_root / "resources" / "models" / "scanner" / args.model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded. Close the overlay window or press Ctrl+C in the console to quit.")

    # Attempt to infer image size from model name
    if args.image_size is None:
        match = re.match(r"img(\d+)_.*", args.model_name)
        if match:
            image_size = int(match.group(1))
        else:
            raise ValueError(f"Could not infer image size from model name: {args.model_name}")
    else:
        image_size = int(args.image_size)
    
    interval = max(0.05, float(args.interval))
    threshold = float(args.threshold)
    debug = args.debug

    app = QApplication([])

    overlay = Overlay()

    last_time = 0.0

    def tick() -> None:
        nonlocal last_time

        now = time.time()
        if now - last_time < interval:
            return
        last_time = now

        # Hide overlay so it does not appear in the screenshot
        # overlay.hide()
        # Process the hide event to ensure the window is not drawn
        QApplication.processEvents()

        frame = grab_screenshot_bgr(debug=debug)
        if frame is None:
            logger.debug("No Camera Frame Ready")
            return
        
        height, width = frame.shape[:2]
        results = model(frame, imgsz=image_size)
        detections = extract_detections(results, threshold)

        # Scale from screenshot pixel space to overlay window space
        dst_w = overlay.width()
        dst_h = overlay.height()
        scaled_detections = scale_detections(detections, width, height, dst_w, dst_h)


        if detections:
            print("Found detections:")
            for det in detections:
                print(f"  {det}")
        # Show overlay again with updated detections
        # overlay.show()
        overlay.update_detections(scaled_detections, margin=10)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(10)

    def handle_sigint(sig, frame):  # type: ignore[override]
        print("!!!!!!!!!!!!!! SHUTTING DOWN !!!!!!!!!!!!!!!!")
        timer.stop()
        app.quit()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        app.exec()
    finally:
        timer.stop()


if __name__ == "__main__":
    main()
