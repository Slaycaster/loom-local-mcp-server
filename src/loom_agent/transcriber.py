"""Audio transcription using OpenAI Whisper."""

import logging
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Error during audio transcription."""
    pass


class Transcriber:
    """Transcribe audio from video files using OpenAI Whisper."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, video_path: str) -> list[dict]:
        """
        Transcribe audio from a video file.

        Args:
            video_path: Path to video file (Whisper handles audio extraction)

        Returns:
            List of segment dicts with start, end, text keys

        Raises:
            TranscriptionError: If transcription fails
        """
        try:
            model = self._get_model()
            logger.info(f"Transcribing: {video_path}")
            result = model.transcribe(video_path)
            return result["segments"]
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}")

    def format_srt_timestamp(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def segments_to_srt(self, segments: list[dict]) -> str:
        """Convert Whisper segments to SRT format string."""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self.format_srt_timestamp(seg["start"])
            end = self.format_srt_timestamp(seg["end"])
            text = seg["text"].strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def write_srt(self, segments: list[dict], output_path: Path) -> None:
        """Write segments to an SRT file."""
        srt_content = self.segments_to_srt(segments)
        output_path.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT file written to: {output_path}")
