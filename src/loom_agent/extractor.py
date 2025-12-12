# src/loom_agent/extractor.py
"""Frame extraction using ffmpeg scene detection."""

import re
import uuid
import subprocess
from pathlib import Path
from typing import Any


class ExtractionError(Exception):
    """Error during frame extraction."""
    pass


class FrameExtractor:
    """Extract frames from video using ffmpeg scene detection."""

    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)

    def create_output_dir(self, video_identifier: str) -> Path:
        """Create a unique output directory for this extraction."""
        # Use UUID to ensure uniqueness
        unique_id = f"{video_identifier}_{uuid.uuid4().hex[:8]}"
        output_dir = self.output_base_dir / unique_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as M:SS or MM:SS timestamp."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def parse_showinfo_line(self, line: str) -> dict[str, Any] | None:
        """Parse a showinfo filter output line to extract pts_time."""
        # Match pts_time:XX.XX pattern
        match = re.search(r'pts_time:\s*(\d+\.?\d*)', line)
        if match:
            return {"pts_time": float(match.group(1))}
        return None

    def apply_max_frames(self, frames: list[dict], max_frames: int) -> list[dict]:
        """Limit frames to max_frames, keeping evenly distributed selection."""
        if len(frames) <= max_frames:
            return frames

        if max_frames <= 2:
            # Just return first and last
            return [frames[0], frames[-1]][:max_frames]

        # Always include first and last, distribute rest evenly
        result = [frames[0]]

        # Calculate step for middle frames
        middle_count = max_frames - 2
        step = (len(frames) - 2) / (middle_count + 1)

        for i in range(1, middle_count + 1):
            idx = int(i * step)
            result.append(frames[idx])

        result.append(frames[-1])
        return result

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise ExtractionError(f"Could not probe video: {result.stderr}")
            return float(result.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired) as e:
            raise ExtractionError(f"Error getting video duration: {e}")

    def extract_frames(
        self,
        video_path: str,
        output_dir: Path,
        threshold: float = 0.3,
        max_frames: int = 20,
        timeout: int = 120
    ) -> list[dict]:
        """
        Extract frames at scene changes using ffmpeg.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            threshold: Scene change threshold (0.0-1.0)
            max_frames: Maximum frames to extract
            timeout: Extraction timeout in seconds

        Returns:
            List of frame info dicts with path, timestamp, scene_score
        """
        output_pattern = str(output_dir / "frame_%03d.png")

        # ffmpeg command with scene detection and showinfo
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfn",
            output_pattern,
            "-y"  # Overwrite output files
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # ffmpeg outputs to stderr
            stderr = result.stderr

        except subprocess.TimeoutExpired:
            raise ExtractionError(f"Frame extraction timed out after {timeout} seconds")
        except FileNotFoundError:
            raise ExtractionError("ffmpeg not found. Check Docker container.")

        # Parse showinfo output to get timestamps
        timestamps = []
        for line in stderr.split('\n'):
            if 'showinfo' in line.lower() or 'pts_time' in line:
                parsed = self.parse_showinfo_line(line)
                if parsed:
                    timestamps.append(parsed["pts_time"])

        # Find all extracted frame files
        frame_files = sorted(output_dir.glob("frame_*.png"))

        if not frame_files:
            # No scenes detected - extract a single frame at the start
            self._extract_single_frame(video_path, output_dir / "frame_001.png")
            frame_files = [output_dir / "frame_001.png"]
            timestamps = [0.0]

        # Build frame info list
        frames = []
        for i, frame_path in enumerate(frame_files):
            timestamp = timestamps[i] if i < len(timestamps) else 0.0

            # Calculate duration until next frame
            if i < len(frame_files) - 1 and i + 1 < len(timestamps):
                duration = timestamps[i + 1] - timestamp
                duration_str = self.format_timestamp(duration)
            else:
                duration_str = None

            frames.append({
                "path": str(frame_path),
                "timestamp": self.format_timestamp(timestamp),
                "scene_score": threshold,  # Simplified: actual score parsing would need more complex ffmpeg output
                "duration_until_next": duration_str
            })

        # Apply max frames limit
        return self.apply_max_frames(frames, max_frames)

    def _extract_single_frame(self, video_path: str, output_path: Path) -> None:
        """Extract a single frame from the start of the video."""
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vframes", "1",
            str(output_path),
            "-y"
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
