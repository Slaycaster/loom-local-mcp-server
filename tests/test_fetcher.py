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
