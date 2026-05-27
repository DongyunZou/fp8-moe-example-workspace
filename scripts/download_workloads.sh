#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/data"

if [[ -z "${WORKLOAD_ARCHIVE_URL:-}" ]]; then
  cat >&2 <<'EOF'
WORKLOAD_ARCHIVE_URL is not set.

Set it to a .tar, .tar.gz, .tgz, .tar.zst, or .zip archive containing the
expected data/ layout, then run this script again.

For a dry run, use:

  python3 verify.py --workloads examples/sample_workloads.jsonl

EOF
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

archive="$tmp/workloads"
curl -L "$WORKLOAD_ARCHIVE_URL" -o "$archive"

case "$WORKLOAD_ARCHIVE_URL" in
  *.tar.zst)
    tar --zstd -xf "$archive" -C "$repo_root/data"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "$archive" -C "$repo_root/data"
    ;;
  *.tar)
    tar -xf "$archive" -C "$repo_root/data"
    ;;
  *.zip)
    unzip -q "$archive" -d "$repo_root/data"
    ;;
  *)
    echo "Unsupported archive extension: $WORKLOAD_ARCHIVE_URL" >&2
    exit 2
    ;;
esac

echo "Workloads extracted under $repo_root/data"

