#!/bin/zsh
# Start Speak Coach locally, then open http://127.0.0.1:8765
cd "$(dirname "$0")"
command -v espeak-ng >/dev/null || brew install espeak-ng
command -v ffmpeg >/dev/null || brew install ffmpeg
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
exec uv run uvicorn app.server:app --host 127.0.0.1 --port "${PORT:-8765}" ${RELOAD:+--reload}
