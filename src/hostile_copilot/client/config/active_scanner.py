from pathlib import Path
from hostile_copilot.config.app_config import AppConfig, Bind


class ActiveScannerConfig(AppConfig):
    # Run options
    monitor = Bind[int]("app.capture.monitor")
    fps = Bind[float]("app.fps")
    confidence = Bind[float]("app.confidence")

    # Overlay settings
    overlay_enabled = Bind[bool]("app.overlay.enabled")
    overlay_show_detections = Bind[bool]("app.overlay.show_detections")
    overlay_monitor = Bind[int]("app.overlay.monitor")

    # Preview settings
    preview_enabled = Bind[bool]("app.preview.enabled")
    preview_monitor = Bind[int]("app.preview.monitor")

    # Inference Models
    device = Bind[str]("app.ai.device")
    detector_model = Bind[Path]("app.ai.detector.model", converter=Path)
    reader_model = Bind[Path]("app.ai.reader.model", converter=Path)

    # Training Options - Extracting data for future training
    save = Bind[bool]("app.training.save")
    save_dir = Bind[Path]("app.training.save_dir", converter=Path)
    all_labels = Bind[bool]("app.training.all_labels")
    labels = Bind[list[str]]("app.training.labels", arg_key="label", action="append")
    min_capture_interval = Bind[float]("app.training.min_capture_interval", arg_key="capture_interval")

    # Logging outputs
    verbose = Bind[bool]("app.verbose")
    debug = Bind[bool]("app.debug.enable")
    profiler = Bind[bool]("app.debug.profiler")
    profiler_dump_interval = Bind[int]("app.debug.profiler_dump_interval")

    # Frame Inspection Options
    inspect_frame = Bind[bool]("app.debug.inspect.frame.enable")
    inspect_frame_path = Bind[Path]("app.debug.inspect.frame.file_path", converter=Path)
    inspect_frame_interval = Bind[int]("app.debug.inspect.frame.interval")
 
    inspect_digit_reader_frame = Bind[bool]("app.debug.inspect.digit_reader_frame.enable")
    inspect_digit_reader_frame_path = Bind[Path]("app.debug.inspect.digit_reader_frame.file_path", converter=Path)
    inspect_digit_reader_frame_interval = Bind[int]("app.debug.inspect.digit_reader_frame.interval")

    inspect_ping_detection = Bind[bool]("app.debug.inspect.ping_detection.enable")
    inspect_ping_detection_path = Bind[Path]("app.debug.inspect.ping_detection.file_path", converter=Path, default="screenshots/last_ping_detection.png")
    inspect_ping_detection_interval = Bind[int]("app.debug.inspect.ping_detection.interval")