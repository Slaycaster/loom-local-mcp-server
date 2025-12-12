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
