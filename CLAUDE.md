# Loom Local MCP Server

## Overview

MCP (Model Context Protocol) server that extracts key frames from Loom videos or local video files for visual debugging analysis. Runs as a Docker container, invoked via `docker exec` by Claude Code using stdio transport.

## Stack

- **Language:** Python 3.11
- **Build system:** Hatchling (`pyproject.toml`)
- **Framework:** FastMCP (`mcp` package) for MCP server
- **Video processing:** ffmpeg (scene detection + frame extraction), ffprobe (duration)
- **Audio transcription:** OpenAI Whisper (base model, CPU inference, lazy-loaded)
- **Video downloading:** yt-dlp (Loom, YouTube, and other URLs)
- **Models:** Pydantic v2 for request/response schemas
- **Testing:** pytest + pytest-asyncio
- **Containerization:** Docker + Docker Compose

## Project Structure

```
src/loom_agent/
  server.py      # MCP server definition, extract_video_frames tool
  fetcher.py     # URL detection, local file lookup, yt-dlp download
  extractor.py   # ffmpeg scene-detection frame extraction + interval fallback
  transcriber.py # Whisper audio transcription + SRT output
  models.py      # Pydantic models (FrameInfo, TranscriptSegment, ExtractionResponse)
  __main__.py    # Entry point
tests/           # pytest tests for each module + integration
```

## MCP Tool

**`extract_video_frames`** — single tool exposed via MCP:
- `source`: Loom/YouTube URL or local filename (from `~/loom-videos/`)
- `threshold`: Scene change sensitivity (0.0-1.0, default 0.3)
- `max_frames`: Max frames to extract (default 20)
- `include_transcript`: Transcribe audio via Whisper (default false)
- Returns frame paths (PNG), timestamps, scene scores, and optionally transcript segments + SRT file

## Key Behaviors

- Scene detection via ffmpeg `select='gt(scene,threshold)'`; falls back to interval-based extraction when <3 frames detected (common for screen recordings)
- 30-minute video duration limit
- Downloaded videos are cleaned up after extraction
- Frame paths in responses use host-accessible paths (via `LOOM_FRAMES_HOST_DIR` env var) so Claude Code can read them
- Whisper model is lazy-loaded on first transcription request; model size configurable via `WHISPER_MODEL` env var (default: `base`)

## Development

```bash
pip install -e ".[dev]"   # Install with dev deps
pytest                     # Run tests
python -m loom_agent       # Run server locally
./scripts/setup.sh         # Build + start Docker container
```
