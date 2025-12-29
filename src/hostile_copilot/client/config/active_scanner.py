from hostile_copilot.config.app_config import Bindings, Bind

class ActiveScannerBindings(Bindings):
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
    detector_model = Bind[str]("app.ai.detector.model")
    reader_model = Bind[str]("app.ai.reader.model")

    # Training Options - Extracting data for future training
    save = Bind[bool]("app.training.save")
    save_dir = Bind[str]("app.training.save_dir")
    all_labels = Bind[bool]("app.training.all_labels")
    labels = Bind[list[str]]("app.training.labels", arg_key="label", action="append")
    min_capture_interval = Bind[float]("app.training.min_capture_interval", arg_key="capture_interval")

    # Logging outputs
    verbose = Bind[bool]("app.verbose")
    debug = Bind[bool]("app.debug.enable")
    profiler = Bind[bool]("app.debug.profiler")

    # Frame Inspection Options
    inspect_frame = Bind[bool]("app.debug.inspect.frame.enable")
    inspect_frame_path = Bind[str]("app.debug.inspect.frame.file_path")
    inspect_frame_interval = Bind[int]("app.debug.inspect.frame.interval")
    inspect_digit_reader_frame = Bind[bool]("app.debug.inspect.digit_reader_frame.enable")
    inspect_digit_reader_frame_path = Bind[str]("app.debug.inspect.digit_reader_frame.file_path")
    inspect_digit_reader_frame_interval = Bind[int]("app.debug.inspect.digit_reader_frame.interval")
