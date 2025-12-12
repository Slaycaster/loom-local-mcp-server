# tests/test_integration.py
"""Integration tests - require ffmpeg and optionally network access."""

import pytest
import tempfile
import subprocess
import os
from pathlib import Path

# Check if ffmpeg is available
FFMPEG_AVAILABLE = subprocess.run(
    ["which", "ffmpeg"], capture_output=True
).returncode == 0


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
class TestIntegration:
    """Integration tests that require ffmpeg."""

    def test_extract_from_local_file(self):
        """Test full extraction pipeline with a generated test video."""
        from loom_agent.extractor import FrameExtractor

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate a simple test video with scene changes
            test_video = Path(tmpdir) / "test.mp4"
            output_dir = Path(tmpdir) / "frames"
            output_dir.mkdir()

            # Create a 3-second test video with color changes (simulates scenes)
            # Red for 1s, then green for 1s, then blue for 1s
            cmd = [
                "ffmpeg",
                "-f", "lavfi",
                "-i", "color=red:duration=1:size=320x240:rate=30",
                "-f", "lavfi",
                "-i", "color=green:duration=1:size=320x240:rate=30",
                "-f", "lavfi",
                "-i", "color=blue:duration=1:size=320x240:rate=30",
                "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
                str(test_video),
                "-y"
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Extract frames
            extractor = FrameExtractor(output_base_dir=str(output_dir))
            sub_dir = extractor.create_output_dir("test")
            frames = extractor.extract_frames(
                video_path=str(test_video),
                output_dir=sub_dir,
                threshold=0.3,
                max_frames=10
            )

            # Should have extracted at least 1 frame
            assert len(frames) >= 1

            # Each frame should have required fields
            for frame in frames:
                assert "path" in frame
                assert "timestamp" in frame
                assert Path(frame["path"]).exists()
