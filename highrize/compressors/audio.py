"""
Audio compressor — converts audio to transcript text for LLMs.

Strategy:
  1. Remove silence (optional, reduces transcription cost)
  2. Transcribe using Whisper (local) or OpenAI Whisper API
  3. Return transcript text — far fewer tokens than raw audio embedding

This is the most cost-effective audio strategy: instead of embedding
audio tokens, you get plain text which compresses further with TextCompressor.
"""

import os
import io
import tempfile
from typing import Optional, Union
from ..models import CompressionResult, Modality
from .text import TextCompressor, _estimate_tokens


class AudioCompressor:
    """
    Compresses audio by transcribing it to text.

    Args:
        backend: "whisper_local" (requires `openai-whisper` pip package)
                 "whisper_api"   (requires openai API key)
                 "faster_whisper" (requires `faster-whisper` pip package, faster on CPU)
        model_size: Whisper model size for local — "tiny", "base", "small", "medium", "large"
        language: Force language code e.g. "en", "hi". None = auto-detect.
        remove_silence: Use pydub to strip silent segments before transcription.
        compress_transcript: Run TextCompressor on the result.
        openai_api_key: Required if backend="whisper_api"
    """

    def __init__(
        self,
        token_counter=None,
        backend: str = "whisper_local",
        model_size: str = "base",
        language: Optional[str] = None,
        remove_silence: bool = True,
        compress_transcript: bool = True,
        openai_api_key: Optional[str] = None,
    ):
        from ..tokens import TokenCounter
        self.backend = backend
        self.model_size = model_size
        self.language = language
        self.remove_silence = remove_silence
        self.compress_transcript = compress_transcript
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._whisper_model = None
        self.token_counter = token_counter or TokenCounter()

    def compress(self, audio_path: str) -> CompressionResult:
        # Estimate raw audio "tokens" (speech LLMs like Whisper charge per second ~40 tokens/sec)
        duration_sec = self._get_duration(audio_path)
        original_tokens = int(duration_sec * 40)

        processed_path = audio_path
        temp_file = None
        if self.remove_silence:
            temp_file = self._strip_silence(audio_path)
            if temp_file != audio_path:
                processed_path = temp_file

        transcript = self._transcribe(processed_path)

        if temp_file and temp_file != audio_path:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        if self.compress_transcript:
            tc = TextCompressor(token_counter=self.token_counter)
            result = tc.compress(transcript)
            transcript = result.compressed

        compressed_tokens = self.token_counter.count(transcript)

        return CompressionResult(
            original=audio_path,
            compressed=transcript,
            modality=Modality.AUDIO,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            metadata={
                "duration_seconds": duration_sec,
                "backend": self.backend,
                "model": self.model_size,
                "language": self.language,
                "silence_removed": self.remove_silence,
            },
        )

    def _get_duration(self, audio_path: str) -> float:
        try:
            import wave
            with wave.open(audio_path, "rb") as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        except Exception:
            # Fallback for non-WAV
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0
            except Exception:
                return 60.0  # Assume 1 min if unknown

    def _strip_silence(self, audio_path: str) -> str:
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence

            audio = AudioSegment.from_file(audio_path)
            chunks = split_on_silence(
                audio,
                min_silence_len=500,
                silence_thresh=audio.dBFS - 16,
                keep_silence=100,
            )
            if not chunks:
                return audio_path

            combined = chunks[0]
            if len(chunks) > 1:
                for chunk in chunks[1:]:
                    combined += chunk

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            combined.export(tmp.name, format="wav")
            return tmp.name
        except (ImportError, IndexError):
            return audio_path

    def _transcribe(self, audio_path: str) -> str:
        if self.backend == "whisper_local":
            return self._transcribe_local(audio_path)
        elif self.backend == "whisper_api":
            return self._transcribe_api(audio_path)
        elif self.backend == "faster_whisper":
            return self._transcribe_faster(audio_path)
        else:
            raise ValueError(f"Unknown audio backend: {self.backend}")

    def _transcribe_local(self, audio_path: str) -> str:
        try:
            import whisper
        except ImportError:
            raise ImportError("Install openai-whisper: pip install openai-whisper")

        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model(self.model_size)

        result = self._whisper_model.transcribe(
            audio_path,
            language=self.language,
            fp16=False,
        )
        return result["text"].strip()

    def _transcribe_faster(self, audio_path: str) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Install faster-whisper: pip install faster-whisper")

        if self._whisper_model is None:
            self._whisper_model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

        segments, _ = self._whisper_model.transcribe(audio_path, language=self.language)
        return " ".join(seg.text for seg in segments).strip()

    def _transcribe_api(self, audio_path: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=self.openai_api_key)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=self.language,
            )
        return result.text.strip()
