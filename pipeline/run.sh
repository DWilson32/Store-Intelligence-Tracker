#!/usr/bin/env bash
# run.sh — Process all CCTV clips for all stores and emit events to JSONL
#
# Usage:
#   ./pipeline/run.sh --clips_dir /data/clips --layout /data/store_layout.json \
#                     --output_dir /data/events

set -euo pipefail

# Delegate to store-agnostic Python pipeline runner
if command -v python3 &>/dev/null; then
  python3 pipeline/run_pipeline.py "$@"
else
  python pipeline/run_pipeline.py "$@"
fi

