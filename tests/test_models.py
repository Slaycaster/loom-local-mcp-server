import pytest
from loom_agent.models import FrameInfo, ExtractionResponse


def test_frame_info_creation():
    frame = FrameInfo(
        path="/tmp/frames/frame_001.png",
        timestamp="0:12",
        scene_score=0.45,
        duration_until_next="0:08"
    )
    assert frame.path == "/tmp/frames/frame_001.png"
    assert frame.timestamp == "0:12"
    assert frame.scene_score == 0.45
    assert frame.duration_until_next == "0:08"


def test_extraction_response_success():
    response = ExtractionResponse(
        status="success",
        video_duration="2:34",
        frames_extracted=2,
        frames=[
            FrameInfo(
                path="/tmp/frames/frame_001.png",
                timestamp="0:00",
                scene_score=0.95,
                duration_until_next="0:12"
            )
        ],
        message="Extracted 2 frames"
    )
    assert response.status == "success"
    assert response.frames_extracted == 2
    assert len(response.frames) == 1


def test_extraction_response_error():
    response = ExtractionResponse(
        status="error",
        message="Could not fetch video"
    )
    assert response.status == "error"
    assert response.frames_extracted == 0
    assert response.frames == []
