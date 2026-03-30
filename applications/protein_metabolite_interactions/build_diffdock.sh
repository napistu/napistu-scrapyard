#!/usr/bin/env bash
# Simple DiffDock Docker build script for ARM64 (Mac M-series)

set -e

IMAGE_TAG="${1:-diffdock:cpu}"

echo "Building DiffDock Docker image for ARM64..."
echo "Image tag: $IMAGE_TAG"
echo ""
echo "This will take 10-20 minutes and download ~5GB"
echo ""

docker build \
  --platform linux/arm64 \
  -t "$IMAGE_TAG" \
  -f Dockerfile.diffdock.arm64 \
  .

echo ""
echo "✓ Build complete: $IMAGE_TAG"
echo ""
echo "Test with: docker run --rm -it $IMAGE_TAG --help"