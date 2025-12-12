# tests/test_extractor.py
import pytest
import tempfile
from pathlib import Path
from loom_agent.extractor import FrameExtractor, ExtractionError


def test_extractor_init():
    extractor = FrameExtractor(output_base_dir="/tmp/frames")
    assert extractor.output_base_dir == Path("/tmp/frames")


def test_create_output_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        extractor = FrameExtractor(output_base_dir=tmpdir)
        output_dir = extractor.create_output_dir("test_video")
        assert output_dir.exists()
        assert output_dir.parent == Path(tmpdir)


def test_format_timestamp():
    extractor = FrameExtractor(output_base_dir="/tmp")
    assert extractor.format_timestamp(0) == "0:00"
    assert extractor.format_timestamp(62.5) == "1:02"
    assert extractor.format_timestamp(3661) == "61:01"


def test_parse_showinfo_line():
    extractor = FrameExtractor(output_base_dir="/tmp")

    # Sample ffmpeg showinfo output line
    line = "[Parsed_showinfo_1 @ 0x...] n:   0 pts:   1234 pts_time:12.34 ..."

    result = extractor.parse_showinfo_line(line)
    assert result is not None
    assert result["pts_time"] == 12.34


def test_parse_showinfo_line_invalid():
    extractor = FrameExtractor(output_base_dir="/tmp")
    result = extractor.parse_showinfo_line("random line without pts_time")
    assert result is None


def test_apply_max_frames_limit():
    extractor = FrameExtractor(output_base_dir="/tmp")

    # Create 10 mock frames
    frames = [{"path": f"/tmp/frame_{i:03d}.png", "timestamp": f"0:{i:02d}"} for i in range(10)]

    # Limit to 5 frames - should keep evenly distributed
    limited = extractor.apply_max_frames(frames, max_frames=5)
    assert len(limited) == 5
    # Should include first and last
    assert limited[0]["path"] == "/tmp/frame_000.png"
    assert limited[-1]["path"] == "/tmp/frame_009.png"


def test_apply_max_frames_no_limit_needed():
    extractor = FrameExtractor(output_base_dir="/tmp")
    frames = [{"path": f"/tmp/frame_{i}.png"} for i in range(3)]
    limited = extractor.apply_max_frames(frames, max_frames=10)
    assert len(limited) == 3
