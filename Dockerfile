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
