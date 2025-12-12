# Loom Agent MCP Server Design

## Overview

A Python-based MCP server that extracts key frames from video content (Loom URLs or local files) for debugging analysis in Claude Code. Uses ffmpeg scene detection to capture meaningful state changes.

**Primary use case:** Code debugging - extracting relevant screens (error messages, code snippets, UI states) from Loom videos showing bugs.

## Architecture

### Core Flow

```
Input (URL/file) → yt-dlp (if URL) → ffmpeg scene detection → PNG frames → output directory → return paths + metadata
```

### Components

| File | Purpose |
|------|---------|
| `server.py` | FastMCP-based server exposing the extraction tool |
| `fetcher.py` | Handles yt-dlp for URLs, validates local files |
| `extractor.py` | ffmpeg scene detection + frame extraction |
| `models.py` | Pydantic models for frame metadata and responses |

### Project Structure

```
loom-agent/
├── scripts/
│   └── setup.sh          # Build and start script
├── src/
│   └── loom_agent/
│       ├── __init__.py
│       ├── server.py
│       ├── fetcher.py
│       ├── extractor.py
│       └── models.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Docker Setup

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN pip install yt-dlp

# Install Python dependencies
COPY pyproject.toml .
RUN pip install .

# Create temp directory for frames
RUN mkdir -p /tmp/loom-frames

COPY src/ ./src/

CMD ["python", "-m", "loom_agent.server"]
```

### docker-compose.yml

```yaml
services:
  loom-agent:
    build: .
    container_name: loom-agent
    restart: unless-stopped
    volumes:
      - ~/loom-videos:/videos:ro       # inbox for local files (read-only)
      - ~/loom-frames:/tmp/loom-frames # output frames (host-accessible)
```

### Claude Code MCP Config

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "loom-agent": {
      "command": "docker",
      "args": ["exec", "-i", "loom-agent", "python", "-m", "loom_agent.server"]
    }
  }
}
```

## MCP Tool Interface

### Tool: `extract_video_frames`

```python
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
    """
```

### Response Format

```python
{
    "status": "success",
    "video_duration": "2:34",
    "frames_extracted": 8,
    "frames": [
        {
            "path": "/home/user/loom-frames/abc123/frame_001.png",
            "timestamp": "0:00",
            "scene_score": 0.95,
            "duration_until_next": "0:12"
        },
        {
            "path": "/home/user/loom-frames/abc123/frame_002.png",
            "timestamp": "0:12",
            "scene_score": 0.45,
            "duration_until_next": "0:08"
        }
    ],
    "message": "Extracted 8 key frames from 2:34 video. Frames saved to /home/user/loom-frames/abc123/"
}
```

## Frame Extraction Logic

### Scene Detection

Uses ffmpeg's built-in `select` filter for scene change detection:

```python
import ffmpeg

def extract_frames(video_path: str, output_dir: str, threshold: float = 0.3):
    """
    Extract frames at scene changes using ffmpeg.

    threshold: 0.0-1.0, maps to ffmpeg's scene detection
               0.3 = moderate sensitivity (good default)
               0.1 = very sensitive (many frames)
               0.5 = less sensitive (fewer frames)
    """
    (
        ffmpeg
        .input(video_path)
        .filter('select', f'gt(scene,{threshold})')
        .filter('showinfo')  # outputs timestamp + scene score
        .output(f'{output_dir}/frame_%03d.png', vsync='vfn')
        .run(capture_stderr=True)
    )
```

### Process Flow

1. **Input validation** - Check if source is URL or file path
2. **Download (if URL)** - yt-dlp fetches video to temp location
3. **Create output dir** - `/tmp/loom-frames/{unique_id}/`
4. **Run ffmpeg** - Scene detection + frame extraction
5. **Parse ffmpeg output** - Extract timestamps and scene scores from stderr
6. **Apply max_frames cap** - If too many frames, keep evenly distributed subset
7. **Build response** - Assemble metadata for each frame
8. **Cleanup** - Remove downloaded video (keep frames)

### Image Format

PNG for pixel-perfect text quality in debugging screenshots.

## Error Handling

| Scenario | Detection | Response |
|----------|-----------|----------|
| Invalid Loom URL | yt-dlp returns error | `{"status": "error", "message": "Could not fetch video. Check URL is public and valid."}` |
| Local file not found | `os.path.exists()` check | `{"status": "error", "message": "File not found: filename"}` |
| Unsupported format | ffmpeg probe fails | `{"status": "error", "message": "Unsupported video format"}` |
| No scenes detected | Zero frames extracted | Return single frame at 0:00 + warning message |
| ffmpeg not available | subprocess error | `{"status": "error", "message": "ffmpeg not found. Check Docker container."}` |
| Video too long (>30min) | Duration check | `{"status": "error", "message": "Video exceeds 30min limit. Use local file with trimmed clip."}` |

### Timeouts

- yt-dlp download: 5 minute timeout
- ffmpeg extraction: 2 minute timeout

### Logging

Errors logged to stderr, viewable via:
```bash
docker logs loom-agent
```

## Setup Script

### scripts/setup.sh

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

## Usage

### For Loom URLs

In Claude Code:
> "I'm working on this bug in `auth.py`. Here's the Loom showing it: https://loom.com/share/abc123"

Claude Code will:
1. Call `extract_video_frames` with the URL
2. Read the extracted frames from `~/loom-frames/`
3. Help debug using the visual context

### For Local Files

1. Drop video file in `~/loom-videos/bug-recording.mp4`
2. In Claude Code:
   > "Here's a local video showing the bug: bug-recording.mp4"

## Dependencies

- `mcp` - MCP Python SDK
- `yt-dlp` - Video downloading
- `ffmpeg-python` - Python bindings for ffmpeg
- `pydantic` - Data validation
- System: `ffmpeg` (installed in Docker image)
