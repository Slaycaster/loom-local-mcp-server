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
