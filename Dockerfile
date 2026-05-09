# syntax=docker/dockerfile:1
#
# ARM64 image (Apple Silicon / linux/arm64).
# Build context must be the PARENT directory containing both sibling repos:
#
#   docker build \
#     --platform linux/arm64 \
#     -f propresenter-speech/Dockerfile \
#     -t propresenter-speech:latest \
#     ..
#
# Use build-docker.sh for a convenience wrapper.

FROM --platform=linux/arm64 python:3.12-slim

# portaudio19-dev  PortAudio headers + runtime required by sounddevice
# libpulse0        PulseAudio client libs for audio routing on macOS Docker
# libgomp1         OpenMP runtime required by CTranslate2 (faster-whisper)
# gcc              Fallback compiler for any wheels without ARM prebuilts
RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev \
        libpulse0 \
        libgomp1 \
        gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==1.8.3"

# Both packages live under /app so the path dependency resolves:
#   pyproject.toml: propresenter-slides = {path = "../propresenter-slides"}
#   /app/propresenter-speech/../propresenter-slides  →  /app/propresenter-slides  ✓
WORKDIR /app
COPY propresenter-slides/ propresenter-slides/
COPY propresenter-speech/ propresenter-speech/

WORKDIR /app/propresenter-speech

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN poetry install --only main

# Mount a named volume here to persist Whisper model weights across runs:
#   docker run -v whisper-models:/root/.cache/huggingface propresenter-speech:latest ...
VOLUME ["/root/.cache/huggingface"]

ENTRYPOINT ["propresenter-speech"]
