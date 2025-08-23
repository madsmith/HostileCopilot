import logging
import numpy as np
import pyaudio
import torch
import threading
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Any

from hostile_copilot.audio import AudioData
from hostile_copilot.config import OmegaConfig

logger = logging.getLogger(__name__)

class STTEngineLocal:
    def __init__(self, config: OmegaConfig):
        self._config: OmegaConfig = config
        self._device: torch.device | None = None
        self._processor: WhisperProcessor | None = None
        self._model: WhisperForConditionalGeneration | None = None

        self._lock = threading.Lock()
    
    @property
    def device(self):
        assert self._device is not None, "Device not initialized"
        return self._device
    
    @property
    def processor(self):
        assert self._processor is not None, "Processor not initialized"
        return self._processor
    
    @property
    def model(self):
        assert self._model is not None, "Model not initialized"
        return self._model

    def initialize(self):
        self._initialize_device()
        self._initialize_model()

        self._warmup()

    def _initialize_device(self):
        self._device = torch.device(
            "mps" if torch.mps.is_available()
              else "cuda" if torch.cuda.is_available() 
              else "cpu"
        )
        print("Deivce is " + str(self._device))

    def _initialize_model(self):
        assert self._device is not None, "Failed to initialize device for STT engine"

        try:
            processor_model_id: str = self._config.get("stt.processor.model_id") or self._config.get("stt.model_id")
            generator_model_id: str = self._config.get("stt.generator.model_id") or self._config.get("stt.model_id")

            assert processor_model_id is not None, "Missing config 'stt.model_id or stt.processor.model_id'"
            assert generator_model_id is not None, "Missing config 'stt.model_id or stt.generator.model_id'"

            model = None

            # First attempt to load the model locally
            for local_only in [True, False]:
                try:
                    self._processor = WhisperProcessor.from_pretrained(
                        processor_model_id,
                        return_attention_mask=True,
                        local_files_only=local_only) # type: ignore

                    model = WhisperForConditionalGeneration.from_pretrained(
                        generator_model_id,
                        local_files_only=local_only)
                    break
                except Exception as e:
                    if local_only:
                        logger.warning(f"Failed to load model locally: {e}")
                    else:
                        raise

            if self._processor is None:
                raise RuntimeError("Failed to initialize STT engine: Processor not initialized")
            if model is None:
                raise RuntimeError("Failed to initialize STT engine: Model not initialized")

            if model.generation_config:
                model.generation_config.forced_decoder_ids = None

            # Send model to device
            self._model = model.to(self._device)

        except Exception as e:
            logger.exception(f"Failed to initialize STT engine: {e}")
            raise

    def _warmup(self):
        logger.info("Warming up STT engine...")
        dummy_input = np.zeros(3000, dtype=np.int16)
        self.infer(AudioData(dummy_input.tobytes(), format=pyaudio.paInt16, channels=1, rate=16000))
        logger.info("STT engine warmed up.")

    def infer(self, audio: AudioData, **inference_params: dict[str, Any]) -> str:
        assert audio.format == pyaudio.paInt16, "Audio format must be int16"
        
        audio_bytes = np.frombuffer(audio.as_bytes(), dtype=np.int16)

        with self._lock:
            # Tokenize the audio inputs
            inputs = self.processor(
                audio_bytes,
                sampling_rate=audio.rate,
                return_tensors="pt").to(self._device)
            
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask if "attention_mask" in inputs else None

            # Perform inference
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    attention_mask=attention_mask,
                    **inference_params)
        # End of lock

        transcription: str = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return transcription