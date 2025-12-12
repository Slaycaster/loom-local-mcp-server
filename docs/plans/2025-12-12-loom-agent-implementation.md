# Loom Agent MCP Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python MCP server that extracts key frames from Loom videos or local files for debugging analysis in Claude Code.

**Architecture:** Docker-containerized Python MCP server using FastMCP. Video fetching via yt-dlp, frame extraction via ffmpeg scene detection. Returns file paths + metadata to host-mounted directory.

**Tech Stack:** Python 3.11, FastMCP, yt-dlp, ffmpeg-python, Pydantic, Docker

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/loom_agent/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "loom-agent"
version = "0.1.0"
description = "MCP server for extracting video frames from Loom URLs"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0.0",
    "ffmpeg-python>=0.2.0",
    "pydantic>=2.0.0",
    "yt-dlp>=2024.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/loom_agent"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create package init**

```python
# src/loom_agent/__init__.py
"""Loom Agent - MCP server for video frame extraction."""

__version__ = "0.1.0"
```

**Step 3: Create directory structure**

Run:
```bash
mkdir -p src/loom_agent tests
touch src/loom_agent/__init__.py
```

**Step 4: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: initialize project structure with pyproject.toml"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `src/loom_agent/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_models.py
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
```

**Step 2: Run test to verify it fails**

Run: `pip install -e ".[dev]" && pytest tests/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'loom_agent.models'"

**Step 3: Write minimal implementation**

```python
# src/loom_agent/models.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/loom_agent/models.py tests/test_models.py
git commit -m "feat: add Pydantic models for frame extraction"
```

---

## Task 3: Video Fetcher (URL Detection & Local File Validation)

**Files:**
- Create: `src/loom_agent/fetcher.py`
- Create: `tests/test_fetcher.py`

**Step 1: Write the failing test**

```python
# tests/test_fetcher.py
import pytest
import tempfile
import os
from pathlib import Path
from loom_agent.fetcher import VideoFetcher, VideoSource


def test_detect_url_source():
    fetcher = VideoFetcher(videos_dir="/videos", temp_dir="/tmp")
    source = fetcher.detect_source("https://www.loom.com/share/abc123")
    assert source == VideoSource.URL


def test_detect_local_source():
    fetcher = VideoFetcher(videos_dir="/videos", temp_dir="/tmp")
    source = fetcher.detect_source("video.mp4")
    assert source == VideoSource.LOCAL


def test_detect_loom_share_url():
    fetcher = VideoFetcher(videos_dir="/videos", temp_dir="/tmp")
    source = fetcher.detect_source("https://loom.com/share/abc123def456")
    assert source == VideoSource.URL


def test_validate_local_file_not_found():
    fetcher = VideoFetcher(videos_dir="/nonexistent", temp_dir="/tmp")
    with pytest.raises(FileNotFoundError, match="File not found"):
        fetcher.get_local_path("missing.mp4")


def test_validate_local_file_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = Path(tmpdir) / "test.mp4"
        test_file.touch()

        fetcher = VideoFetcher(videos_dir=tmpdir, temp_dir="/tmp")
        path = fetcher.get_local_path("test.mp4")
        assert path == str(test_file)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'loom_agent.fetcher'"

**Step 3: Write minimal implementation**

```python
# src/loom_agent/fetcher.py
"""Video fetching and source detection."""

import os
import re
import tempfile
import subprocess
from enum import Enum
from pathlib import Path


class VideoSource(Enum):
    URL = "url"
    LOCAL = "local"


class FetchError(Exception):
    """Error fetching video from URL."""
    pass


class VideoFetcher:
    """Handles video source detection, local file validation, and URL downloading."""

    # Patterns that indicate a URL
    URL_PATTERNS = [
        r'^https?://',
        r'^www\.',
        r'loom\.com',
        r'youtube\.com',
        r'youtu\.be',
    ]

    def __init__(self, videos_dir: str, temp_dir: str):
        self.videos_dir = Path(videos_dir)
        self.temp_dir = Path(temp_dir)

    def detect_source(self, source: str) -> VideoSource:
        """Detect whether source is a URL or local filename."""
        for pattern in self.URL_PATTERNS:
            if re.search(pattern, source, re.IGNORECASE):
                return VideoSource.URL
        return VideoSource.LOCAL

    def get_local_path(self, filename: str) -> str:
        """Get full path for a local file in the videos directory."""
        full_path = self.videos_dir / filename
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {filename} (looked in {self.videos_dir})")
        return str(full_path)

    def download_url(self, url: str, timeout: int = 300) -> str:
        """
        Download video from URL using yt-dlp.
        Returns path to downloaded video file.

        Args:
            url: Video URL (Loom, YouTube, etc.)
            timeout: Download timeout in seconds (default 5 minutes)

        Raises:
            FetchError: If download fails
        """
        # Create temp file for output
        output_template = str(self.temp_dir / "download_%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--format", "best[ext=mp4]/best",
            "--output", output_template,
            "--print", "after_move:filepath",
            url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise FetchError(f"Could not fetch video. Check URL is public and valid. Error: {result.stderr}")

            # yt-dlp prints the final filepath
            downloaded_path = result.stdout.strip().split('\n')[-1]

            if not os.path.exists(downloaded_path):
                raise FetchError(f"Download completed but file not found: {downloaded_path}")

            return downloaded_path

        except subprocess.TimeoutExpired:
            raise FetchError(f"Download timed out after {timeout} seconds")
        except FileNotFoundError:
            raise FetchError("yt-dlp not found. Ensure it is installed.")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetcher.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/loom_agent/fetcher.py tests/test_fetcher.py
git commit -m "feat: add video fetcher with URL detection and local validation"
```

---

## Task 4: Frame Extractor (ffmpeg Scene Detection)

**Files:**
- Create: `src/loom_agent/extractor.py`
- Create: `tests/test_extractor.py`

**Step 1: Write the failing test**

```python
# tests/test_extractor.py
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractor.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'loom_agent.extractor'"

**Step 3: Write minimal implementation**

```python
# src/loom_agent/extractor.py
"""Frame extraction using ffmpeg scene detection."""

import os
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractor.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/loom_agent/extractor.py tests/test_extractor.py
git commit -m "feat: add frame extractor with ffmpeg scene detection"
```

---

## Task 5: MCP Server

**Files:**
- Create: `src/loom_agent/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'loom_agent.server'"

**Step 3: Write minimal implementation**

```python
# src/loom_agent/server.py
"""MCP server for video frame extraction."""

import os
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from loom_agent.models import FrameInfo, ExtractionResponse
from loom_agent.fetcher import VideoFetcher, VideoSource, FetchError
from loom_agent.extractor import FrameExtractor, ExtractionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment or defaults
VIDEOS_DIR = os.environ.get("LOOM_VIDEOS_DIR", "/videos")
FRAMES_DIR = os.environ.get("LOOM_FRAMES_DIR", "/tmp/loom-frames")
MAX_VIDEO_DURATION = 30 * 60  # 30 minutes in seconds

# Initialize MCP server
mcp = FastMCP("loom-agent")

# Initialize components
fetcher = VideoFetcher(videos_dir=VIDEOS_DIR, temp_dir="/tmp")
extractor = FrameExtractor(output_base_dir=FRAMES_DIR)


@mcp.tool()
async def extract_video_frames(
    source: str,
    threshold: float = 0.3,
    max_frames: int = 20
) -> dict:
    """
    Extract key frames from a Loom video URL or local video file
    for visual debugging and analysis.

    Args:
        source: Loom URL or local file path (for local files,
                drop in ~/loom-videos and provide filename only)
        threshold: Scene change sensitivity (0.0-1.0).
                   Lower = more frames, higher = fewer frames. Default 0.3
        max_frames: Maximum frames to extract as safety cap. Default 20

    Returns:
        Dictionary with status, frames list, and metadata
    """
    # Validate parameters
    if not 0.0 <= threshold <= 1.0:
        return ExtractionResponse(
            status="error",
            message=f"Invalid threshold: {threshold}. Must be between 0.0 and 1.0"
        ).model_dump()

    if max_frames < 1:
        return ExtractionResponse(
            status="error",
            message=f"Invalid max_frames: {max_frames}. Must be at least 1"
        ).model_dump()

    video_path = None
    downloaded = False

    try:
        # Detect source type
        source_type = fetcher.detect_source(source)
        logger.info(f"Processing {source_type.value} source: {source}")

        if source_type == VideoSource.LOCAL:
            video_path = fetcher.get_local_path(source)
        else:
            # Download from URL
            logger.info(f"Downloading video from URL...")
            video_path = fetcher.download_url(source)
            downloaded = True

        # Check video duration
        duration = extractor.get_video_duration(video_path)
        if duration > MAX_VIDEO_DURATION:
            return ExtractionResponse(
                status="error",
                message=f"Video exceeds 30min limit ({int(duration/60)} minutes). Use local file with trimmed clip."
            ).model_dump()

        # Create output directory
        video_id = Path(source).stem if source_type == VideoSource.LOCAL else "loom"
        output_dir = extractor.create_output_dir(video_id)
        logger.info(f"Extracting frames to {output_dir}")

        # Extract frames
        frames_data = extractor.extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            threshold=threshold,
            max_frames=max_frames
        )

        # Convert to FrameInfo models
        frames = [FrameInfo(**f) for f in frames_data]

        # Build response
        response = ExtractionResponse(
            status="success",
            video_duration=extractor.format_timestamp(duration),
            frames_extracted=len(frames),
            frames=frames,
            message=f"Extracted {len(frames)} key frames from {extractor.format_timestamp(duration)} video. Frames saved to {output_dir}/"
        )

        logger.info(f"Successfully extracted {len(frames)} frames")
        return response.model_dump()

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ExtractionResponse(
            status="error",
            message=str(e)
        ).model_dump()

    except FetchError as e:
        logger.error(f"Fetch error: {e}")
        return ExtractionResponse(
            status="error",
            message=str(e)
        ).model_dump()

    except ExtractionError as e:
        logger.error(f"Extraction error: {e}")
        return ExtractionResponse(
            status="error",
            message=str(e)
        ).model_dump()

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return ExtractionResponse(
            status="error",
            message=f"Unexpected error: {str(e)}"
        ).model_dump()

    finally:
        # Cleanup downloaded video
        if downloaded and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Cleaned up downloaded video: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup video: {e}")


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/loom_agent/server.py tests/test_server.py
git commit -m "feat: add MCP server with extract_video_frames tool"
```

---

## Task 6: Package Entry Point

**Files:**
- Create: `src/loom_agent/__main__.py`

**Step 1: Create entry point**

```python
# src/loom_agent/__main__.py
"""Entry point for running the server as a module."""

from loom_agent.server import main

if __name__ == "__main__":
    main()
```

**Step 2: Test module execution**

Run: `python -c "from loom_agent.server import mcp; print('Server loads OK')"`
Expected: "Server loads OK"

**Step 3: Commit**

```bash
git add src/loom_agent/__main__.py
git commit -m "feat: add module entry point for server"
```

---

## Task 7: Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp (latest version)
RUN pip install --no-cache-dir yt-dlp

# Copy and install Python package
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --no-cache-dir .

# Create directories for volumes
RUN mkdir -p /videos /tmp/loom-frames

# Set environment variables
ENV LOOM_VIDEOS_DIR=/videos
ENV LOOM_FRAMES_DIR=/tmp/loom-frames

# Run the MCP server
CMD ["python", "-m", "loom_agent"]
```

**Step 2: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for containerized deployment"
```

---

## Task 8: Docker Compose

**Files:**
- Create: `docker-compose.yml`

**Step 1: Create docker-compose.yml**

```yaml
services:
  loom-agent:
    build: .
    container_name: loom-agent
    restart: unless-stopped
    volumes:
      # Inbox for local video files (read-only)
      - ~/loom-videos:/videos:ro
      # Output directory for extracted frames (read-write)
      - ~/loom-frames:/tmp/loom-frames
    environment:
      - LOOM_VIDEOS_DIR=/videos
      - LOOM_FRAMES_DIR=/tmp/loom-frames
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add docker-compose for easy container management"
```

---

## Task 9: Setup Script

**Files:**
- Create: `scripts/setup.sh`

**Step 1: Create setup script**

```bash
#!/bin/bash
set -e

echo "=== Loom Agent Setup ==="

# Create host directories
echo "Creating directories..."
mkdir -p ~/loom-videos
mkdir -p ~/loom-frames

# Stop and remove existing container/volumes
echo "Cleaning up existing containers..."
docker compose down -v --remove-orphans 2>/dev/null || true

# Build fresh image
echo "Building image..."
docker compose build --no-cache

# Start container
echo "Starting container..."
docker compose up -d

echo ""
echo "=== Setup complete ==="
echo "Inbox folder: ~/loom-videos (drop local videos here)"
echo "Output folder: ~/loom-frames (frames appear here)"
echo ""
echo "Container status:"
docker compose ps
```

**Step 2: Make executable and test syntax**

Run: `mkdir -p scripts && chmod +x scripts/setup.sh && bash -n scripts/setup.sh && echo "Syntax OK"`
Expected: "Syntax OK"

**Step 3: Commit**

```bash
git add scripts/setup.sh
git commit -m "feat: add setup script for Docker deployment"
```

---

## Task 10: README

**Files:**
- Create: `README.md`

**Step 1: Create README**

```markdown
# Loom Agent

MCP server for extracting key frames from Loom videos or local files for debugging analysis.

## Quick Start

```bash
# Build and start the container
./scripts/setup.sh
```

## Usage

### Claude Code Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "loom-agent": {
      "command": "docker",
      "args": ["exec", "-i", "loom-agent", "python", "-m", "loom_agent"]
    }
  }
}
```

### Analyzing Loom Videos

In Claude Code, simply mention a Loom URL:

> "I'm debugging this issue in auth.py. Here's the Loom showing the bug: https://loom.com/share/abc123"

Claude will automatically extract key frames and analyze them.

### Local Video Files

1. Drop video in `~/loom-videos/`
2. Reference by filename:

> "Check out the bug in this recording: bug-demo.mp4"

## Tool Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `source` | required | Loom URL or local filename |
| `threshold` | 0.3 | Scene change sensitivity (0.0-1.0). Lower = more frames |
| `max_frames` | 20 | Maximum frames to extract |

## Directories

- `~/loom-videos/` - Drop local videos here
- `~/loom-frames/` - Extracted frames appear here

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run server locally (without Docker)
python -m loom_agent
```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Task 11: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Create integration test (skipped without ffmpeg)**

```python
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
```

**Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (or SKIPPED if ffmpeg not installed)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for frame extraction"
```

---

## Task 12: Final Verification

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

**Step 2: Build Docker image**

Run: `docker compose build`
Expected: Successful build

**Step 3: Verify project structure**

Run: `find . -type f -name "*.py" -o -name "*.toml" -o -name "*.yml" -o -name "*.sh" -o -name "*.md" -o -name "Dockerfile" | grep -v __pycache__ | sort`

Expected:
```
./Dockerfile
./docker-compose.yml
./docs/plans/2025-12-12-loom-agent-design.md
./docs/plans/2025-12-12-loom-agent-implementation.md
./pyproject.toml
./README.md
./scripts/setup.sh
./src/loom_agent/__init__.py
./src/loom_agent/__main__.py
./src/loom_agent/extractor.py
./src/loom_agent/fetcher.py
./src/loom_agent/models.py
./src/loom_agent/server.py
./tests/test_extractor.py
./tests/test_fetcher.py
./tests/test_integration.py
./tests/test_models.py
./tests/test_server.py
```

**Step 4: Final commit**

```bash
git add -A
git status
# If any uncommitted files, commit them
git commit -m "chore: final cleanup and verification" --allow-empty
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project scaffolding | `pyproject.toml`, `src/loom_agent/__init__.py` |
| 2 | Pydantic models | `models.py`, `test_models.py` |
| 3 | Video fetcher | `fetcher.py`, `test_fetcher.py` |
| 4 | Frame extractor | `extractor.py`, `test_extractor.py` |
| 5 | MCP server | `server.py`, `test_server.py` |
| 6 | Package entry point | `__main__.py` |
| 7 | Dockerfile | `Dockerfile` |
| 8 | Docker Compose | `docker-compose.yml` |
| 9 | Setup script | `scripts/setup.sh` |
| 10 | README | `README.md` |
| 11 | Integration tests | `test_integration.py` |
| 12 | Final verification | - |
