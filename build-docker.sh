#!/usr/bin/env bash
# Build the propresenter-speech Docker image.
#
# Usage (from the propresenter-speech directory):
#   ./build-docker.sh                        # arm64, tag propresenter-speech:latest
#   ./build-docker.sh propresenter-speech:v1 # custom tag
#   PLATFORM=linux/amd64 DOCKERFILE=Dockerfile.amd64 ./build-docker.sh propresenter-speech:amd64
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

PLATFORM="${PLATFORM:-linux/arm64}"
DOCKERFILE="${DOCKERFILE:-$SCRIPT_DIR/Dockerfile}"
TAG="${1:-propresenter-speech:latest}"

# Temporarily place .dockerignore at the parent-directory level so Docker
# picks it up (Docker reads .dockerignore from the root of the build context).
IGNORE_SRC="$SCRIPT_DIR/.dockerignore"
IGNORE_DST="$PARENT_DIR/.dockerignore"
_cleanup() { [[ -f "$IGNORE_DST" ]] && rm -f "$IGNORE_DST"; }
trap _cleanup EXIT
cp "$IGNORE_SRC" "$IGNORE_DST"

echo "Building $TAG  (platform: $PLATFORM)"
echo "Context: $PARENT_DIR"
docker build \
  --platform "$PLATFORM" \
  -f "$DOCKERFILE" \
  -t "$TAG" \
  "$PARENT_DIR"

echo ""
echo "Build complete. Quick-start:"
echo ""
echo "  # Test CLI flags (no mic, no ProPresenter needed)"
echo "  docker run --rm $TAG --help"
echo "  docker run --rm $TAG --list-devices"
echo ""
echo "  # Run against ProPresenter on this Mac (use host.docker.internal, not localhost)"
echo "  docker run --rm \\"
echo "    -v whisper-models:/root/.cache/huggingface \\"
echo "    -e PULSE_SERVER=host.docker.internal \\"
echo "    $TAG --host host.docker.internal"
echo ""
echo "  See README.md → Docker section for audio (microphone) setup."
