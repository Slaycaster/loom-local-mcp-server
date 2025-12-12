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
