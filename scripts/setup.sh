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
