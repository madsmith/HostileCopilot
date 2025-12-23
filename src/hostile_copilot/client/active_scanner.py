from __future__ import annotations
import ctypes
ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)

# Suppress QT DPI warnings
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

# Load version info
from hostile_copilot import __version__

import argparse
import asyncio
from collections import defaultdict
import cv2
from dataclasses import dataclass
from datetime import datetime
import dxcam
from dxcam import DXFactory
import logging
import math
import numpy as np
from pathlib import Path
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QApplication
import qasync
import signal
import sys
import time
import torch
from typing import Any, Callable
from ultralytics import YOLO

from hostile_copilot.scan_reader import CRNNLoader, CRNN, get_crnn_transform, DecodeUtils
from hostile_copilot.client.components.ui.overlay import Overlay
from hostile_copilot.client.components.ui.components import LabeledBox, Polygon
from hostile_copilot.utils.debug.profiler import Profiler
from hostile_copilot.utils.geometry import BoundingBoxLike, OrientedBoundingBox, is_overlapping

logger = logging.getLogger(__name__)

DETECTOR_MODEL_PATH = Path("resources/models/detector/")
READER_MODEL_PATH = Path("resources/models/digit_reader/")

shutdown_requested = asyncio.Event()

DEBUG = True
VERBOSE = False

def output(*args):
    """
    Wrapper around calling print() that can be disabled with a flag
    """
    to_str = repr if VERBOSE else str
    msg = " ".join(to_str(a) for a in args)

    if VERBOSE:
        logger.info(msg)
    else:
        print(msg)

def _resolve_model_path(model_path: Path, default_path: Path) -> Path:
    if not model_path.exists():
        model_path = Path(default_path) / model_path
    if not model_path.exists():
        model_path = Path(default_path) / model_path.name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path


def load_detector_model(model_path: Path, default_path: Path, device: torch.device) -> YOLO:
    logger.debug(f"Loading detector model from {model_path}")
    model_path = _resolve_model_path(model_path, default_path)
    return YOLO(model_path).to(device)


def load_reader_model(model_path: Path, default_path: Path, device: torch.device) -> CRNN:
    logger.debug(f"Loading reader model from {model_path}")
    model_path = _resolve_model_path(model_path, default_path)
    reader_model = CRNNLoader.from_weights(model_path)
    return reader_model.to(device)


class Camera:
    """
    State wrapper around DXCam camera.
    """
    def __init__(self, screen_number: int | None = None):

        if screen_number is None:
            logger.info("Using primary monitor")
            self.camera = dxcam.create(output_color="BGR")
        else:
            self._select_monitor(screen_number)
            logger.info(f"Using monitor {screen_number} [Device: {self._device_idx}, Output: {self._output_idx}]")
            self.camera: dxcam.DXCamera = dxcam.create(
                device_idx=self._device_idx,
                output_idx=self._output_idx,
                output_color="BGR",
            )
        self._last_frame: np.ndarray | None = None

    def _select_monitor(self, screen_idx: int):
        if hasattr(dxcam, '__factory'):
            factory = getattr(dxcam, '__factory', None)
        else:
            factory = DXFactory()
        count = 0
        for device_idx, outputs in enumerate(factory.outputs):
            for output_idx, output_data in enumerate(outputs):
                if count == screen_idx:
                    self._device_idx = device_idx
                    self._output_idx = output_idx
                    return
                count += 1

        raise IndexError(f"Index {screen_idx} is out of range.")
    
    def capture(self, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        new_frame = self.camera.grab(region)
        if new_frame is None:
            return self._last_frame
        self._last_frame = new_frame
        return new_frame


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bounding_box: BoundingBoxLike

    def __repr__(self):
        return f"DetectedObject(label={self.label}, confidence={self.confidence:.2f}, bounding_box={self.bounding_box})"

    def __str__(self):
        return f"DetectedObject(label={self.label}, confidence={self.confidence:.2f})"


@dataclass
class TrackedObject(DetectedObject):
    id: int
    time_detected: float
    time_seen: float
    count_seen: int = 0
    count_missing: int = 0

    def __repr__(self):
        return f"TrackedObject(label={self.label}, confidence={self.confidence:.2f}, bounding_box={self.bounding_box}, id={self.id}, time_detected={self.time_detected}, time_seen={self.time_seen}, count_seen={self.count_seen}, count_missing={self.count_missing})"

    def __str__(self):
        return f"TrackedObject(label={self.label}, confidence={self.confidence:.2f}, id={self.id})"


class ObjectTracker:
    def __init__(self):
        self._objects: dict[int, TrackedObject] = {}
        self._object_metadata: defaultdict[int, dict[str, Any]] = defaultdict(dict)
        self._next_id = 0


    def update(self, objects: list[DetectedObject]) -> list[TrackedObject]:
        current_time = time.time()

        updated_objects = []
        for target_obj in objects:
            matched = False
            for tracked_obj in self._objects.values():
                if (
                    target_obj.label == tracked_obj.label
                    and is_overlapping(target_obj.bounding_box, tracked_obj.bounding_box)
                ):
                    tracked_obj.bounding_box = target_obj.bounding_box
                    tracked_obj.count_seen += 1
                    tracked_obj.count_missing = 0
                    tracked_obj.time_seen = current_time
                    updated_objects.append(tracked_obj)
                    matched = True
                    break
            
            if not matched:
                tracked_obj = TrackedObject(
                    id=self._next_id,
                    label=target_obj.label,
                    confidence=target_obj.confidence,
                    bounding_box=target_obj.bounding_box,
                    time_detected=current_time,
                    time_seen=current_time,
                    count_seen=1,
                    count_missing=0
                )
                self._objects[self._next_id] = tracked_obj
                updated_objects.append(tracked_obj)
                self._next_id += 1

        # Age objects not seen
        for obj in self._objects.values():
            if obj.time_seen < current_time:
                obj.count_missing += 1
        
        return updated_objects
    
    def expire_missing(self, max_missing: int) -> list[TrackedObject]:
        # Redo with dict delete
        removed_objects = []

        for obj in self._objects.values():
            if obj.count_missing > max_missing:
                removed_objects.append(obj)
        for obj in removed_objects:
            del self._objects[obj.id]
            del self._object_metadata[obj.id]

        return removed_objects
    
    def expire_not_seen(self, max_age: int) -> list[TrackedObject]:
        current_time = time.time()
        removed_objects = []

        for obj in self._objects.values():
            if current_time - obj.time_seen > max_age:
                removed_objects.append(obj)
        for obj in removed_objects:
            del self._objects[obj.id]
            del self._object_metadata[obj.id]

        return removed_objects    

    def get_metadata(self, object_id: int, key: str | None = None) -> dict[str, Any] | Any:
        if key is None:
            return self._object_metadata.get(object_id, {})
        return self._object_metadata.get(object_id, {}).get(key)

    def set_metadata(self, object_id: int, key: str, value: Any) -> None:
        if object_id in self._objects:
            self._object_metadata[object_id][key] = value

    def query(self, criteria: Callable[[TrackedObject, dict[str, Any]], bool]) -> list[TrackedObject]:
        return [obj for idx, obj in self._objects.items() if criteria(obj, self._object_metadata.get(idx, {}))]


class ScanReader:
    def __init__(self, reader_model: CRNN, device: torch.device):
        self._reader_model = reader_model
        self._transforms = get_crnn_transform()
        self._vocabulary = reader_model.config.get("vocabulary")
        self._device = device
        self._inspect_enabled = False
        self._inspect_path = "screenshots/last_reader.png"

        assert self._vocabulary is not None, "Reader model must have an alphabet"

    def enable_inspection(self, path: Path | None = None) -> None:
        self._inspect_enabled = True
        if path is not None:
            self._inspect_path = path

    def predict(self, crop: np.ndarray) -> str:
        crop_rgb = crop[:, :, ::-1]
        pil_image = Image.fromarray(crop_rgb).convert("L")

        img_tensor = self._transforms(pil_image).unsqueeze(0) # (1, H, W) -> (B=1, C=1, H', W')

        img_tensor = img_tensor.to(self._device)

        # Convert back to Image and save
        if self._inspect_enabled:
            preview_img = self._tensor_to_PIL(img_tensor)
            preview_img.save(self._inspect_path)
        
        with torch.no_grad():
            with Profiler("Reader"):
                logits = self._reader_model(img_tensor)
                prediction, confidence = DecodeUtils.greedy_decode_with_confidence(self._vocabulary, logits)
        
        logger.info(f"Reader prediction: {prediction} ({confidence:.2f})")
        return prediction

    @classmethod
    def _tensor_to_PIL(cls, img_tensor: torch.Tensor) -> Image.Image:
        """
        img_tensor: (1, 1, H, W) or (1, H, W) normalized with mean=0.5 std=0.5
        Returns a grayscale PIL Image.
        """
        # Remove batch if present
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]

        # Move to CPU
        img_tensor = img_tensor.detach().cpu()

        # (1, H, W) → (H, W)
        img_tensor = img_tensor.squeeze(0)

        # Unnormalize: [-1,1] → [0,1]
        img_tensor = img_tensor * 0.5 + 0.5

        # Clamp to avoid numeric noise
        img_tensor = img_tensor.clamp(0, 1)

        # Convert to uint8 numpy
        arr = (img_tensor.numpy() * 255).astype(np.uint8)

        # Convert to PIL
        return Image.fromarray(arr, mode="L")

def class_color(class_id: int, total_classes: int = 20, opacity=1.0) -> QColor:
    # distribute hues evenly around the color wheel
    hue = int((class_id % total_classes) * (360 / total_classes))
    sat = 255
    val = 255
    return QColor.fromHsv(hue, sat, val, int(opacity * 255))


def _normalize_whr(width: float, height: float, angle_rad: float, horizontal: bool = True) -> tuple[float, float, float]:
    """
    Normalize OBB so that the long side prefers horizontal (if 
    horizontal=True) or vertical (if horizontal=False).

    Ultralytics OBB angles are in radians. We fold the angle into
    [-90deg, 90deg] and, when it's closer to vertical, swap w/h and
    rotate by +/- 90deg so that the long axis is closer to horizontal
    while preserving the notion of "up".
    """

    # Convert to degrees for easier reasoning
    ang_deg = math.degrees(angle_rad)

    # Fold into [-180, 180)
    ang_deg = (ang_deg + 180.0) % 360.0 - 180.0

    # If closer to vertical (>45deg), rotate by -90 or +90 and swap w/h
    if 45.0 < abs(ang_deg) <= 135.0:
        if ang_deg > 0:
            ang_deg -= 90.0
        else:
            ang_deg += 90.0
        width, height = height, width

    # Finally, clamp to [-90, 90] to avoid large tilts
    if ang_deg > 90.0:
        ang_deg -= 180.0
    elif ang_deg < -90.0:
        ang_deg += 180.0

    return width, height, math.radians(ang_deg)


def convert_to_bounding_box(xywhr: torch.Tensor) -> OrientedBoundingBox:
    x_center, y_center, width, height, angle_rad = xywhr
    width, height, angle_rad = _normalize_whr(width, height, angle_rad)
    return OrientedBoundingBox(x_center, y_center, width, height, angle_rad)
    

def process_frame(frame: np.ndarray, detector_model: YOLO, threshold: float) -> list[DetectedObject]:
    assert frame is not None, "No frame to process"
    with Profiler("detector"):
        results = detector_model(frame, verbose=False)

    objects: list[DetectedObject] = []
    for result in results:
        # Process the Oriented Bounding Box (OBB) results
        obbs = getattr(result, "obb", None)
        if obbs is None:
            continue

        # Move results to CPU and convert to numpy arrays
        class_ids = obbs.cls.cpu().numpy()
        confidences = obbs.conf.cpu().numpy()
        xywhrs = obbs.xywhr.cpu().numpy()

        for class_id, confidence, xywhr in zip(class_ids, confidences, xywhrs):
            class_name = result.names[class_id]

            if confidence < threshold:
                logger.debug(f"Skipping {class_name} with confidence {confidence}")
                continue

            bounding_box: OrientedBoundingBox = convert_to_bounding_box(xywhr)
            objects.append(DetectedObject(class_name, confidence, bounding_box))

    return objects
    

def crop_frame(frame: np.ndarray, bounding_box: BoundingBoxLike, fast: bool = True, overlay: Overlay | None = None) -> np.ndarray | None:
    assert isinstance(bounding_box, OrientedBoundingBox), "Only OrientedBoundingBox is supported"

    obb: OrientedBoundingBox = bounding_box

    src_corners = np.array(obb.corners(), dtype=np.float32)
    # Stabalize by rounding to nearest 2 pixels
    width  = int(round(obb.w / 2) * 2)
    height = int(round(obb.h / 2) * 2)

    if DEBUG and overlay:
        polygon = Polygon(src_corners, color=(255, 0, 0, 255))
        overlay.set_drawables([
            polygon,
        ])

    dst_corners = np.array([
        [0,     0],
        [width, 0],
        [width, height],
        [0,     height],
    ], dtype=np.float32)

    # Transform from source to destination
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    crop = cv2.warpPerspective(
        frame, M, (width, height),
        flags=cv2.INTER_LINEAR if fast else cv2.INTER_CUBIC
    )

    return crop


def process_ping_scans(
    tracked_objects: list[TrackedObject],
    frame: np.ndarray,
    scan_reader: ScanReader,
    tracker: ObjectTracker,
    overlay: Overlay | None = None,
):
    # TODO: debug
    print(f"Processing ping scans: {len(tracked_objects)} objects")
    for obj in tracked_objects:
        if obj.label != "Ping - Scan":
            continue

        # TODO: debug
        print(f"Processing TrackedObject [{obj.id}] {obj.label} - {obj.bounding_box}")

        crop = crop_frame(frame, obj.bounding_box, overlay=overlay)
        if crop is None:
            logger.warning("Fail to detect on missing crop.")
            continue

        previous_text = tracker.get_metadata(obj.id, "text")
        if previous_text:
            tracker.set_metadata(obj.id, "previous_text", previous_text)

        text = scan_reader.predict(crop)
        tracker.set_metadata(obj.id, "text", text)

        if text != previous_text:
            tracker.set_metadata(obj.id, "time_changed", time.time())
        
        # logger.info(f"Detected text: {text}")
        # TODO: show detected text here?
        logger.debug(f"  Processing TrackedObject [{obj.id}] {obj.label} - {obj.bounding_box}")


def process_detection(crop: np.ndarray, detection: DetectedObject, scan_reader: ScanReader, tracker: ObjectTracker) -> None:
    if detection.label != "Ping - Scan":
        logger.debug(f"Skipping {detection.label}")
        return

    if crop is None:
        logger.warning("Fail to detect on missing crop.")
        return

    last_prediction = tracker.get_metadata(detection.id, "text")
    if last_prediction is not None:
        logger.info(f"Last prediction: {last_prediction}")

    text = scan_reader.predict(crop)
    tracker.set_metadata(detection.id, "text", text)

    logger.info(f"Detected text: {text}")


def save_detection(crop: np.ndarray, detection: DetectedObject, args: argparse.Namespace) -> None:
    # Ensure output directory exists
    label_norm = detection.label.replace(" ", "_").lower()
    output_path = Path(args.save_dir) / label_norm
    output_path.mkdir(parents=True, exist_ok=True)

    # Save crop
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{label_norm}_{detection.id}_{timestamp}.png"
    cv2.imwrite(str(output_path / filename), crop)

    # TODO: proper arg to enable this
    if args.inspect:
        print("Saving inspection frame")
        inspect_path = Path(args.inspect_path).parent
        inspect_path.mkdir(parents=True, exist_ok=True)
        inspect_filename = inspect_path / "last_detection.png"
        print(f"Saving to {inspect_filename}")
        cv2.imwrite(str(inspect_filename), crop)


def check_save_detection(crop: np.ndarray, detection: DetectedObject, tracker: ObjectTracker, args: argparse.Namespace) -> None:
    if not args.save:
        return

    now = time.time()
    if args.all_labels or detection.label in args.label:
        last_capture_time = tracker.get_metadata(detection.id, "capture_time")
        
        if last_capture_time is not None and now - last_capture_time < args.capture_interval:
            return
        
        save_detection(crop, detection, args)

        tracker.set_metadata(detection.id, "capture_time", now)


def setup_logging(verbose: bool, debug: bool) -> None:
    # Configure logging output for the app
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    
    logger.addHandler(handler)
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logger.setLevel(level)
    
    # Prevent propagation to root (so other libraries stay quiet)
    logger.propagate = False
    
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active ping scanner for Star Citizen.")

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--monitor", "--mon", "-m", type=int, default=None,
        help="Specify the monitor number to capture from (Count starting at 1, otherwise attempts to detect primary monitor)"
    )
    parser.add_argument(
        "--fps", "-f", type=float, default=1.0,
        help="Target processing rate in frames per second. Default: %(default)s"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.5,
        help="Minimum confidence threshold for detections. Default: %(default)s"
    )

    extraction_group = parser.add_argument_group("Extraction")
    extraction_group.add_argument(
        "--save", "-s", action="store_true",
        help="Save extracted detections to disk"
    )
    extraction_group.add_argument(
        "--save-dir", type=str, default="captures",
        help="Directory in which to save extracted detections. Each detection will be saved to a folder with the label name in the destination folder. Default: %(default)s"
    )
    extraction_group.add_argument(
        "--all-labels", action="store_true",
        help="Save all labels to disk"
    )
    extraction_group.add_argument(
        "--label", "-l", type=str, nargs="+", default=[],
        help="Specify a label to save. Can be specified multiple times."
    )
    extraction_group.add_argument(
        "--capture-interval", type=float, default=3.0,
        help="Minimum time interval between captures for the same object. Default: %(default)s"
    )

    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument("--debug", action="store_true", help="Enable debug output")
    advanced_group.add_argument("--profiler", action="store_true", help="Enable profiler output")
    advanced_group.add_argument(
        "--detector-model", type=str, default="scanner.pt",
        help=f"Path to object detector model. Can also be resolved under the path: {DETECTOR_MODEL_PATH}. Default: %(default)s"
    )
    advanced_group.add_argument(
        "--reader-model", type=str, default="digit_reader.pt",
        help=f"Path to digit reader model. Can also be resolved under the path: {READER_MODEL_PATH}. Default: %(default)s"
    )
    advanced_group.add_argument(
        "--device", "-d", type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
        help="Device to run models on. Default: %(default)s"
    )
    advanced_group.add_argument("--inspect", action="store_true", help="Inspect camera capture by writing frames intermittently to disk.")
    advanced_group.add_argument(
        "--inspect-path", type=str, default="screenshots/last_frame.png",
        help="File path for inspected camera frame. Default: %(default)s"
    )
    advanced_group.add_argument(
        "--inspect-interval", type=int, default=1,
        help="Interval between saves of inspected camera frames in seconds. Default: %(default)s"
    )
    advanced_group.add_argument(
        "--inspect-reader", action="store_true", help="Inspect reader output by writing frames intermittently to disk."
    )
    advanced_group.add_argument(
        "--inspect-reader-path", type=str, default="screenshots/last_reader.png",
        help="File path for inspected reader frame. Default: %(default)s"
    )
    advanced_group.add_argument("--overlay-detections", action="store_true", help="Overlay detections on camera feed")
    advanced_group.add_argument(
        "--overlay-monitor", type=int, default=None,
        help="Monitor index to show the overlay.  Defaults to using the same "
             "index as --monitor but for framework reasons those may differ. "
             "Indexing starts at 1.")

    return parser.parse_args()
    

async def run_app(args: argparse.Namespace):
    setup_logging(args.verbose, args.debug)

    if args.profiler:
        Profiler.enable()
    
    inspect_path = Path(args.inspect_path)
    inspect_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    detector_model: YOLO = load_detector_model(
        Path(args.detector_model),
        default_path=DETECTOR_MODEL_PATH,
        device=device
    )
    reader_model: CRNN = load_reader_model(
        Path(args.reader_model),
        default_path=READER_MODEL_PATH,
        device=device
    )

    scan_reader = ScanReader(reader_model, device)
    if args.inspect_reader:
        scan_reader.enable_inspection(Path(args.inspect_reader_path))

    screens = QGuiApplication.screens()
    overlay_monitor = args.overlay_monitor if args.overlay_monitor is not None else args.monitor
    screen = screens[overlay_monitor - 1]
    overlay = Overlay()
    overlay.showOnScreen(screen)

    if args.monitor is None:
        camera = Camera()
    else:
        monitor_num = args.monitor - 1
        camera = Camera(screen_number=monitor_num)

    tick_interval = 1 / args.fps
    profile_dump_interval = 10
    profile_dump_schedule_time = time.time() + profile_dump_interval
    last_inspect_camera_save = 0

    logger.debug(f"Tick interval: {tick_interval}s")
    frame_count = 0
    tracker = ObjectTracker()

    async def schedule_tick(tick_start: float):
        with Profiler("sleep"):
            now = time.time()
            delay = max(0, tick_start + tick_interval - now)
            if delay > 0:
                await asyncio.sleep(delay)

    def criteria_new_persistent(obj: TrackedObject, metadata: dict[str, Any], persistence_age: float = 0.3) -> bool:
        now = time.time()

        time_updated = metadata.get("time_updated")
        time_processed = metadata.get("time_processed", 0)
        
        last_modify_time = time_updated or obj.time_detected
        persistent_age = now - last_modify_time

        if persistent_age < persistence_age:
            return False

        return time_processed <= last_modify_time


    while not shutdown_requested.is_set():
        tick_start = time.time()

        with Profiler("capture"):
            logger.debug(f"Capturing frame... {frame_count}")
            frame = camera.capture()
            logger.debug(f"Captured frame shape: {frame.shape}")
            frame_count += 1

        if args.inspect and time.time() > (last_inspect_camera_save + args.inspect_interval):
            with Profiler("inspect"):
                logger.debug("Saving frame...")
                cv2.imwrite(str(inspect_path), frame)
                last_inspect_camera_save = time.time()

        if Profiler.is_enabled() and time.time() > profile_dump_schedule_time:
            Profiler.dump()
            profile_dump_schedule_time = time.time() + profile_dump_interval

        objects = process_frame(frame, detector_model, args.confidence)

        tracked_objects = tracker.update(objects)

        process_ping_scans(tracked_objects, frame, scan_reader, tracker, overlay)

        new_detections = tracker.query(criteria_new_persistent)

        for detection in new_detections:
            
            output("Detected new object:", detection)

            crop = crop_frame(frame, detection.bounding_box)

            if args.save:
                check_save_detection(crop, detection, tracker, args)

            tracker.set_metadata(detection.id, "time_processed", time.time())

        if args.overlay_detections:
            overlay.clear_drawables()
            new_drawables = []
            for obj in tracked_objects:
                aabb = obj.bounding_box.to_aabb()
                logger.debug(f"Drawing object: {obj.label} at {aabb}")
                drawable = LabeledBox(
                    x1=aabb[0],
                    y1=aabb[1],
                    x2=aabb[2],
                    y2=aabb[3],
                    label=obj.label,
                    color=QColor("red"),
                    font_opacity=0.5,
                    opacity=0.2
                )
                new_drawables.append(drawable)
            overlay.add_drawables(new_drawables)

        # Check for expired tracking and notify
        expired = tracker.expire_missing(3)
        for obj in expired:
            logger.debug(f"Expired missing object: {obj}")
        expired = tracker.expire_not_seen(0.5)
        for obj in expired:
            logger.debug(f"Expired not seen object: {obj}")

        await schedule_tick(tick_start)
        overlay.clear_drawables()


def main() -> None:
    global VERBOSE
    global DEBUG

    app = QApplication(sys.argv)

    args = parse_args()

    if args.verbose:
        VERBOSE = True
    if args.debug:
        DEBUG = True

    # Setup event loop to communicate with QT
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # Force Python to check signals periodically
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # no-op
    timer.start(100)
    
    # Handle Ctrl+C gracefully
    def on_sigterm(*_):
        logger.debug("Shutdown requested")
        shutdown_requested.set()
        app.quit()
    signal.signal(signal.SIGINT, on_sigterm)

    # Run the event loop
    with loop:
        try:
            loop.run_until_complete(run_app(args))
        except (KeyboardInterrupt, RuntimeError) as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error: {e}")
            pass


if __name__ == "__main__":
    main()
