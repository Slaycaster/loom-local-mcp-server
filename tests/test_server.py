# tests/test_server.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from loom_agent.server import extract_video_frames
from loom_agent.fetcher import VideoSource


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


@pytest.mark.asyncio
async def test_extract_frames_with_transcript():
    """Test that include_transcript triggers transcription."""
    mock_frames = [
        {"path": "/tmp/loom-frames/frame_001.png", "timestamp": "0:00", "scene_score": 0.3, "duration_until_next": "0:05"}
    ]
    mock_segments = [
        {"start": 0.0, "end": 2.0, "text": " Hello world"}
    ]

    with patch("loom_agent.server.fetcher") as mock_fetcher, \
         patch("loom_agent.server.extractor") as mock_extractor, \
         patch("loom_agent.server.transcriber") as mock_transcriber:

        mock_fetcher.detect_source.return_value = VideoSource.LOCAL
        mock_fetcher.get_local_path.return_value = "/videos/test.mp4"
        mock_extractor.get_video_duration.return_value = 60.0
        mock_extractor.create_output_dir.return_value = Path("/tmp/loom-frames/test_abc123")
        mock_extractor.extract_frames.return_value = mock_frames
        mock_extractor.format_timestamp.return_value = "1:00"
        mock_transcriber.transcribe.return_value = mock_segments
        mock_transcriber.write_srt.return_value = None

        result = await extract_video_frames(
            source="test.mp4",
            include_transcript=True
        )

        assert result["status"] == "success"
        assert len(result["transcript"]) == 1
        assert result["transcript"][0]["text"] == "Hello world"
        assert result["transcript_file"] is not None
        mock_transcriber.transcribe.assert_called_once()
