import asyncio
import logging
import numpy as np
import openwakeword
from openwakeword import Model as OpenWakeWordModel
import silero_vad
from silero_vad.model import OnnxWrapper

from hostile_copilot.audio import AudioDevice, AudioBuffer, AudioData, numpy_to_tensor, load_wave_file, load_mp3_file
from hostile_copilot.config import OmegaConfig
from hostile_copilot.tts.engine import TTSEngine
from hostile_copilot.utils.logging import get_trace_logger

from .recording_state import RecordingState, RecState, RecEvent

logger = get_trace_logger(__name__)

class VoiceClient:
    def __init__(self, config: OmegaConfig, audio_device: AudioDevice):
        self._config = config
        self._audio_device = audio_device

        model_id = self._config.get("tts.model_id")
        voices = self._config.get("tts.voices")
        self._tts_engine: TTSEngine = TTSEngine(model_id, voices)

        self._wake_word_model: OpenWakeWordModel | None = None
        self._wake_word_config: dict[str, dict[str, str]] = self._config.get("openwakeword.models")
        self._vad_model: OnnxWrapper | None = None
        # self._whisper_engine: AudioInferenceEngine | None = None

        # parameters for audio processing
        self._vad_chunk_frame_count = 512
        self._wake_word_chunk_frame_count = 1280
        self._vad_activation_threshold = 0.5

        self._wake_word_buffer: AudioBuffer = AudioBuffer()
        self._vad_buffer: AudioBuffer = AudioBuffer()
        self._speech_buffer: AudioBuffer = AudioBuffer()

        self._silence_duration: float = 0.0
        self._recording_state: RecordingState = RecordingState()

        self._confirmation_audios: dict[str, bytes] = {}

        self._tasks: list[asyncio.Task] = []

    async def initialize(self):
        await self._tts_engine.initialize()

        # Initialize wake word models
        self._initialize_wake_word_models()

        # Initialize VAD model
        self._initialize_vad_model()

        # Load confirmation audio
        self._load_confirmation_audio()

    def _initialize_wake_word_models(self):
        if self._wake_word_config == None:
            raise ValueError("Missing config 'openwakeword.models'")

        models: list[str] = [
            model.path if "path" in model else model.name
            for model in self._wake_word_config
        ]

        if len(models) == 0:
            raise ValueError("No models specified in config 'openwakeword.models'")

        inference_framework = self._config.get("openwakeword.inference_framework")

        if inference_framework == None:
            raise ValueError("Missing config 'openwakeword.inference_framework'")
        if inference_framework not in ["onnx", "tflite"]:
            raise ValueError(f"Invalid inference framework: {inference_framework}")

        openwakeword.utils.download_models()
        
        self._wake_word_model = OpenWakeWordModel(
            wakeword_models=models,
            inference_framework=inference_framework,
            enable_speex_noise_suppression=False
        )

    def _load_confirmation_audio(self):
        if self._wake_word_config == None:
            raise ValueError("Missing config 'openwakeword.models'")
        
        for model in self._wake_word_config:
            if "confirmation_audio" in model:
                assert "name" in model, "Missing model name in config 'openwakeword.models'"
                if model["confirmation_audio"].endswith(".mp3"):
                    self._confirmation_audios[model["name"]] = load_mp3_file(model["confirmation_audio"])
                elif model["confirmation_audio"].endswith(".wav"):
                    self._confirmation_audios[model["name"]] = load_wave_file(model["confirmation_audio"])
                else:
                    raise ValueError(f"Invalid confirmation audio format: {model['confirmation_audio']}")

    def _initialize_vad_model(self):
        self._vad_model = silero_vad.load_silero_vad()
        if self._vad_model is None:
            raise RuntimeError("Failed to load VAD model")

    async def run(self):
        assert self._audio_device.running, "Audio device must be running"

        try:
            self.running = True
            while self.running:
                wake_word_needed_frames = self._wake_word_chunk_frame_count - self._wake_word_buffer.frame_count()
                vad_needed_frames = self._vad_chunk_frame_count - self._vad_buffer.frame_count()

                frames_needed = min(wake_word_needed_frames, vad_needed_frames)

                audio_bytes: bytes = self._audio_device.read(frames_needed)
                
                self._wake_word_buffer.append(audio_bytes)
                self._vad_buffer.append(audio_bytes)

                self._process_vad()
                self._process_wake_word()

                # yield to other tasks
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Error in run: {e}")
        finally:
            await self.stop()

    async def stop(self):
        self.running = False

        # Cancel remaining tasks
        for task in self._tasks:
            task.cancel()

    def _process_vad(self):
        assert self._vad_model is not None, "VAD model not initialized"
        
        sample_rate = self._vad_buffer.get_sample_rate()

        while self._vad_buffer.frame_count() >= self._vad_chunk_frame_count:
            audio_frames = self._vad_buffer.get_frames()
            self._vad_buffer.clear()

            # push back any extra frames
            if len(audio_frames) > self._vad_chunk_frame_count:
                extra_frames = audio_frames[self._vad_chunk_frame_count:]
                audio_frames = audio_frames[:self._vad_chunk_frame_count]
                self._vad_buffer.append(extra_frames)

            # silero vad expects a tensor (normalized to -1 to 1)
            torch_audio = numpy_to_tensor(audio_frames, np.int16)

            vad_score = self._vad_model(torch_audio, sample_rate).item()

            is_speech = vad_score > self._vad_activation_threshold

            if is_speech:
                logger.trace(f"VAD score: {vad_score}")
                self._silence_duration = 0
                self._recording_state.on_event(RecEvent.VAD_DETECTED)
            else:
                self._silence_duration += self._vad_chunk_frame_count / sample_rate

            if is_speech or self._recording_state.is_recording():
                self._speech_buffer.append(audio_frames)

            if self._recording_state.state == RecState.SPEECH_PENDING:
                if self._silence_duration > self._config.get("voice.followup_timeout", 5):
                    logger.trace("Followup timeout reached, stopping recording")
                    self.stop_recording(cancel=True)

            if self._silence_duration > self._config.get("voice.speech_timeout", 1.5):
                if self._recording_state.is_recording():
                    if self._recording_state.is_processing_speech():
                        logger.trace("Silence detected, stopping recording")
                        self.stop_recording()
                        
                else:
                    logger.trace("Silence detected, clearing speech buffer")
                    self._speech_buffer.clear()

    def _process_wake_word(self):
        activation_threshold = self._config.get("voice.activation_threshold", 0.5)
        if self._wake_word_buffer.frame_count() >= self._wake_word_chunk_frame_count:
            logger.trace("Processing wake word buffer")
            audio_frames = self._wake_word_buffer.get_frames()
            self._wake_word_buffer.clear()

            # push back any extra frames
            if len(audio_frames) > self._wake_word_chunk_frame_count:
                extra_frames = audio_frames[self._wake_word_chunk_frame_count:]
                audio_frames = audio_frames[:self._wake_word_chunk_frame_count]
                self._wake_word_buffer.append(extra_frames)

            # Discard frames if we are already recording
            if self._recording_state.is_recording():
                return

            # openwakeword expects numpy array
            np_audio = np.frombuffer(audio_frames, dtype=np.int16)

            detection: dict[str, float] = self._wake_word_model.predict(np_audio) # type: ignore

            if any(value > activation_threshold for value in detection.values()):
                detected_wake_words = [key for key, value in detection.items() if value > activation_threshold]
                logger.info(f"\nWake word detected! {detected_wake_words}")
                self._wake_word_model.reset()
                for word in detected_wake_words:
                    logger.debug(f"Detected wake word: {word} VAD: {self._speech_buffer.get_duration_ms()}ms")
                    raw_audio = self._speech_buffer.get_bytes()
                    audio_data = AudioData(
                        data=raw_audio,
                        rate=self._speech_buffer.get_sample_rate(),
                        channels=self._speech_buffer.get_channels(),
                        format=self._speech_buffer.get_audio_format()
                    )
                    task = asyncio.create_task(
                        self._task_confirm_wake_word(word, audio_data),
                        name="VoiceClient::ConfirmWakeWord"
                    )
                    self._tasks.append(task)
                    task.add_done_callback(self._tasks.remove)
                    self.start_recording()

    async def _task_confirm_wake_word(self, wake_word: str, audio_data: AudioData):
        assert isinstance(audio_data, AudioData), f"Expected AudioData, got {type(audio_data)}"
        if not await self._confirm_wake_word(wake_word, audio_data):
            logger.debug(f"Wake word {wake_word} not confirmed")
            self.stop_recording()
        else:
            logger.debug(f"Wake word {wake_word} confirmed")
            print(self._confirmation_audios.keys())
            confirmation_audio = self._confirmation_audios.get(wake_word)
            if confirmation_audio:
                self._audio_device.play(confirmation_audio)
            self.start_recording(confirmed=True)

        return True

    async def _confirm_wake_word(self, wake_word: str, audio_data: AudioData):
        assert isinstance(audio_data, AudioData), f"Expected AudioData, got {type(audio_data)}"
        # TODO: Whisper confirmation

        return True

    def start_recording(self, confirmed=False):
        """
        Begin recording audio for processing.
        
        Args:
            confirmed (bool): The audio is a confirmed activation and doesn't need confirmation that the speaker
            is engaging with the voice client.
        """
        logger.debug("Recording started")
        self._recording_state.start()
        if confirmed:
            self._recording_state.confirm()

    def stop_recording(self, cancel=False):
        """
        Stop recording audio.
        
        Args:
            cancel (bool): Whether the recording was cancelled and the audio should not be processed.
        """
        if cancel:
            logger.debug("Recording cancelled")
            self._recording_state.stop()
        elif not self._recording_state.is_processing_speech():
            logger.debug("Recording not yet confirmed")
            return
        else:
            logger.debug("Recording stopped")
            self._recording_state.stop()
            raw_data = self._speech_buffer.get_bytes()

            audio_data = AudioData(
                data=raw_data,
                format=self._speech_buffer.get_audio_format(),
                channels=self._speech_buffer.get_channels(),
                rate=self._speech_buffer.get_sample_rate(),
            )


            logger.trace(f"Audio data length: {len(audio_data)}")

            if not audio_data:
                logger.warning("No audio data to process")
                return
        
            task = asyncio.create_task(
                self._process_recording(audio_data),
                name="VoiceClient::ProcessRecording"
            )
            self._tasks.append(task)
            task.add_done_callback(self._tasks.remove)
    
    async def _process_recording(self, audio_data: AudioData):
        assert isinstance(audio_data, AudioData), f"Expected AudioData, got {type(audio_data)}"
        print("He we heard some audio!!")