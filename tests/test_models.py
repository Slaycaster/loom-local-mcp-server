import pytest
from loom_agent.models import FrameInfo, ExtractionResponse, TranscriptSegment


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


def test_transcript_segment_creation():
    segment = TranscriptSegment(
        start="0:05",
        end="0:12",
        text="Hello world"
    )
    assert segment.start == "0:05"
    assert segment.end == "0:12"
    assert segment.text == "Hello world"


def test_extraction_response_with_transcript():
    response = ExtractionResponse(
        status="success",
        video_duration="2:34",
        frames_extracted=1,
        frames=[],
        message="Extracted 1 frame",
        transcript=[
            TranscriptSegment(start="0:00", end="0:05", text="Hello")
        ],
        transcript_file="/tmp/loom-frames/test/transcript.srt"
    )
    assert len(response.transcript) == 1
    assert response.transcript_file == "/tmp/loom-frames/test/transcript.srt"


def test_extraction_response_without_transcript():
    response = ExtractionResponse(
        status="success",
        video_duration="1:00",
        frames_extracted=1,
        frames=[],
        message="Extracted 1 frame"
    )
    assert response.transcript == []
    assert response.transcript_file is None


def test_extraction_response_error():
    response = ExtractionResponse(
        status="error",
        message="Could not fetch video"
    )
    assert response.status == "error"
    assert response.frames_extracted == 0
    assert response.frames == []
