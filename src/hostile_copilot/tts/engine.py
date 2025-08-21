import asyncio
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from kokoro import KModel, KPipeline # type: ignore
import logging
import numpy as np
import torch
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "hexgrad/Kokoro-82M"

class TTSEngine:
    def __init__(self, model_id: str | None = None, voices: str | list[str] | None = None, lang_code: str | None = None):
        self._model_id: str = model_id or DEFAULT_MODEL
        self._lang_code: str = lang_code or "en-US"

        if isinstance(voices, str):
            voices = [voices]
        elif voices is None:
            voices = []
        
        self._voices: list[str] = voices

        self._device: torch.device | None = None
        self._model: KModel | None = None
        self._pipeline: KPipeline | None = None

        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def pipeline(self) -> KPipeline:
        if self._pipeline is None:
            raise ValueError("TTS engine is not initialized.")
        return self._pipeline
    
    @property
    def model(self) -> KModel:
        if self._model is None:
            raise ValueError("TTS engine is not initialized.")
        return self._model
    
    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise ValueError("TTS engine is not initialized.")
        return self._device
    
    async def initialize(self):
        logger.info("Initializing TTS engine...")
        # Initialize torch device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        logger.info(f"Loading TTS model {self._model_id}...")
        try:
            self._model = KModel(self._model_id).to(self._device)
        except RepositoryNotFoundError as e:
            logger.error(
                "Hugging Face repo not found or inaccessible: %s.\n"
                "- Check the repo id is correct (current: '%s').\n"
                "- If the model is private/gated, authenticate with Hugging Face (set HF_TOKEN env var or run 'huggingface-cli login').\n"
                "- Or set a different public model via config key 'tts.model_id'.",
                e, self._model_id,
            )
            raise
        except HfHubHTTPError as e:
            logger.error(
                "Failed to download model from Hugging Face for '%s': %s.\n"
                "You may need to authenticate or choose a different public model via 'tts.model_id'.",
                self._model_id, e,
            )
            raise

        logger.debug(f"Loading TTSpipeline...")
        self._pipeline = KPipeline(
            lang_code=self._lang_code,
            repo_id=self._model_id,
            model=self._model,
            device=self._device)

        logger.info("TTS Preload voices...")
        for voice in self._voices:
            logger.debug(f"Preloading voice {voice}...")
            self._pipeline.load_voice(voice)

        logger.info("TTS engine initialized.")

    async def infer(self, text: str, inference_params: dict[str, Any] | None = None) -> bytes:
        async with self._lock:
            processed_inputs: str = self._preprocess_input(text)

            logger.debug(f"Generating speech for '{processed_inputs}'...")
            inference_params = inference_params or {}

            if "voices" not in inference_params:
                assert len(self._voices) > 0, "No voices specified."
                inference_params["voice"] = self._voices[0]
            inference_params["text"] = processed_inputs

            generator = self.pipeline(**inference_params)

            result_audio = []
            for i, (grapheme_stream, phoneme_stream, audio) in enumerate(generator):
                # print(f"Generated spech {i}: {gs} -> {ps}")
                result_audio.append(audio)

            logger.debug(f"Generated {len(result_audio)} audio chunks.")
            audio = torch.cat(result_audio, dim=0)

            # copy back
            audio = audio.cpu()

            audio_np: np.ndarray = audio.numpy()
            audio_bytes: bytes = audio_np.tobytes()
            
            return audio_bytes
    
    def _preprocess_input(self, text: str) -> str:
        """
        Run any general preprocessing on the input text
        """
        return text