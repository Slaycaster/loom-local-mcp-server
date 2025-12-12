# tests/test_server.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from loom_agent.server import extract_video_frames


@pytest.mark.asyncio
async def test_extract_frames_invalid_threshold():
    """Test that invalid threshold returns error."""
    result = await extract_video_frames(
        source="test.mp4",
        threshold=1.5,  # Invalid: > 1.0
        max_frames=10
    )
    assert result["status"] == "error"
    assert "threshold" in result["message"].lower()


@pytest.mark.asyncio
async def test_extract_frames_invalid_max_frames():
    """Test that invalid max_frames returns error."""
    result = await extract_video_frames(
        source="test.mp4",
        threshold=0.3,
        max_frames=0  # Invalid: must be > 0
    )
    assert result["status"] == "error"
    assert "max_frames" in result["message"].lower()
