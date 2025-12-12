"""Pydantic models for frame extraction responses."""

from pydantic import BaseModel


class FrameInfo(BaseModel):
    """Metadata for a single extracted frame."""
    path: str
    timestamp: str
    scene_score: float
    duration_until_next: str | None = None


class ExtractionResponse(BaseModel):
    """Response from the frame extraction tool."""
    status: str  # "success" or "error"
    video_duration: str | None = None
    frames_extracted: int = 0
    frames: list[FrameInfo] = []
    message: str
