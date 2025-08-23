import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import openwakeword
from openwakeword import Model as OpenWakeWordModel
import silero_vad
from silero_vad.model import OnnxWrapper
from typing import Callable, Awaitable, Union

from hostile_copilot.audio import AudioDevice, AudioBuffer, AudioData, numpy_to_tensor, load_wave_file, load_mp3_file
from hostile_copilot.config import OmegaConfig
from hostile_copilot.tts import TTSEngineLoader, TTSEngine
from hostile_copilot.stt import STTEngineLoader, STTEngine
from hostile_copilot.utils.logging import get_trace_logger

from .recording_state import RecordingState, RecState, RecEvent

logger = get_trace_logger(__name__)

CallbackT = Callable[[str], Union[None, Awaitable[None]]]

class VoiceClient:
    def __init__(self, config: OmegaConfig, audio_device: AudioDevice):
        self._config = config
        self._audio_device = audio_device

        self._tts_engine: TTSEngine = TTSEngineLoader(config).load()

        self._stt_engine: STTEngine = STTEngineLoader(config).load()

        self._wake_word_model: OpenWakeWordModel | None = None
        self._wake_word_config: dict[str, dict[str, str]] = {}
        for model in self._config.get("openwakeword.models"):
            self._wake_word_config[model.get("name")] = model

        self._vad_model: OnnxWrapper | None = None
        # self._whisper_engine: AudioInferenceEngine | None = None

        self._prompt_callback: CallbackT | None = None
        self._immediate_activation_callback: CallbackT | None = None

        # parameters for audio processing
        self._vad_chunk_frame_count = 512
        self._wake_word_chunk_frame_count = 1280
        self._vad_activation_threshold = 0.5

        self._wake_word_buffer: AudioBuffer = AudioBuffer()
        self._vad_buffer: AudioBuffer = AudioBuffer()
        self._speech_buffer: AudioBuffer = AudioBuffer()

        self._silence_duration: float = 0.0
        self._recording_state: RecordingState = RecordingState()

        self._confirmation_audios: dict[str, str] = {}
        self._audio_resources: dict[str, bytes] = {}

        self._bg_futures: set[Future] = set()
        self._bg_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="VoiceClient::"
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self):
        await self._tts_engine.initialize()
        self._stt_engine.initialize()

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
            for model in self._wake_word_config.values()
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
        
        for model in self._wake_word_config.values():
            if "confirmation_audio" in model:
                assert "name" in model, "Missing model name in config 'openwakeword.models'"

                resource_name = model["confirmation_audio"]

                if not resource_name in self._audio_resources:
                    if resource_name.endswith(".mp3"):
                        self._audio_resources[resource_name] = load_mp3_file(resource_name)
                    elif resource_name.endswith(".wav"):
                        self._audio_resources[resource_name] = load_wave_file(resource_name)
                    else:
                        raise ValueError(f"Invalid confirmation audio format: {model['confirmation_audio']}")
                
                self._confirmation_audios[model["name"]] = resource_name

    def _initialize_vad_model(self):
        self._vad_model = silero_vad.load_silero_vad(onnx=True)
        if self._vad_model is None:
            raise RuntimeError("Failed to load VAD model")

    async def run(self):
        assert self._audio_device.running, "Audio device must be running"

        try:
            self._loop = asyncio.get_running_loop()
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

        # Shutdown background executor
        if self._bg_executor is not None:
            self._bg_executor.shutdown(wait=False, cancel_futures=True)
            self._bg_executor = None
        self._bg_futures.clear()

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
                logger.info(f"Wake word detected! {detected_wake_words}")
                self._wake_word_model.reset()
                self._on_wake_words_detected(detected_wake_words)

    def _confirm_wake_word_sync(self, wake_word: str, audio_data: AudioData) -> bool:
        assert isinstance(audio_data, AudioData), f"Expected AudioData, got {type(audio_data)}"
        # TODO: Whisper confirmation (CPU-heavy) goes here synchronously
        # Return True if confirmed, False otherwise
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
        
            self._submit_bg(self._process_recording_async, audio_data)
    
    def on_prompt(self, callback: CallbackT):
        self._prompt_callback = callback
            
    def on_immediate_activation(self, callback: CallbackT):
        self._immediate_activation_callback = callback
    
    def _process_recording_async(self, audio_data: AudioData) -> None:
        assert isinstance(audio_data, AudioData), f"Expected AudioData, got {type(audio_data)}"
        print("He we heard some audio!!")

        import time
        start = time.time()
        try:
            text = self._stt_engine.infer(audio_data)
        except Exception as e:
            logger.exception(f"STT inference failed: {e}")
            return
        end = time.time()
        print("Text: " + text)
        print("Time: " + str(end - start))

        if self._prompt_callback:
            try:
                if asyncio.iscoroutinefunction(self._prompt_callback):
                    fut = asyncio.run_coroutine_threadsafe(self._prompt_callback(text), self._loop)
                    fut.result()
                else:
                    self._prompt_callback(text)
            except Exception as e:
                logger.exception(f"Prompt callback failed: {e}")

    def _process_immediate_wake_activation(self, wake_word: str) -> None:
        assert isinstance(wake_word, str), f"Expected str, got {type(wake_word)}"
        if self._immediate_activation_callback:
            try:
                if asyncio.iscoroutinefunction(self._immediate_activation_callback):
                    fut = asyncio.run_coroutine_threadsafe(self._immediate_activation_callback(wake_word), self._loop)
                    fut.result()
                else:
                    self._immediate_activation_callback(wake_word)
            except Exception as e:
                logger.exception(f"Immediate activation callback failed: {e}")
        
    def _on_wake_words_detected(self, words: list[str]) -> None:
        """Schedule confirmation for detected wake words without blocking audio loop."""
        if not words:
            return
        scheduled = False
        for word in words:
            logger.debug(f"Detected wake word: {word} VAD: {self._speech_buffer.get_duration_ms()}ms")

            raw_audio = self._speech_buffer.get_bytes()
            audio_data = AudioData(
                data=raw_audio,
                rate=self._speech_buffer.get_sample_rate(),
                channels=self._speech_buffer.get_channels(),
                format=self._speech_buffer.get_audio_format()
            )
            self._submit_bg(
                self._confirm_wake_word_sync,
                word,
                audio_data,
                on_done=lambda f, w=word: self._confirm_done(f, w),
            )
            scheduled = True

        if scheduled:
            # Begin recording immediately while confirmation runs
            self.start_recording()

    def _confirm_done(self, f: Future, wake_word: str) -> None:
        try:
            confirmed = f.result()
        
            if not confirmed:
                logger.debug(f"Wake word {wake_word} not confirmed")
                self.stop_recording()
            else:
                logger.debug(f"Wake word {wake_word} confirmed")
                resource_name = self._confirmation_audios.get(wake_word)
                if resource_name:
                    assert resource_name in self._audio_resources, f"Missing audio resource: {resource_name}"
                    self._audio_device.play(self._audio_resources[resource_name])

                if self._wake_word_config[wake_word].get("suppress_follow_up", False):
                    # TODO: raise event to notify UI
                    self.stop_recording(cancel=True)
                    self._submit_bg(self._process_immediate_wake_activation, wake_word)
                    return

                self.start_recording(confirmed=True)
        except Exception as e:
            logger.exception(f"Confirm wake word failed: {e}")
            self.stop_recording(cancel=True)

    def _submit_bg(self, fn, *args, on_done=None) -> Future | None:
        """Submit a function to the background executor and optionally schedule an on_done callback back on the event loop."""
        if self._bg_executor is None:
            return None
        fut: Future = self._bg_executor.submit(fn, *args)
        self._bg_futures.add(fut)

        def _cb(f: Future):
            self._bg_futures.discard(f)
            if on_done is not None and self._loop is not None:
                # marshal callback to the asyncio loop thread, but ensure exceptions don't kill the loop
                def _safe_on_done():
                    try:
                        on_done(f)
                    except Exception as e:
                        logger.exception(f"Background completion handler failed: {e}")
                self._loop.call_soon_threadsafe(_safe_on_done)

        fut.add_done_callback(_cb)
        return fut