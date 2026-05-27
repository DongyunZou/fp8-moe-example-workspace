#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 verify.py --workloads examples/sample_workloads.jsonl --dump-json runs/smoke.json

