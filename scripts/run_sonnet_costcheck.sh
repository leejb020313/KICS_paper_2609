#!/usr/bin/env bash
# Re-call Claude Sonnet 5 on a sample of test batches in clean CLI mode only to log latency and token usage
# (the September scoring runs kept only the text reply). Scores from these calls are not used anywhere.
# Usage (from results/frontier/): xargs -P 3 -n 1 bash ../../scripts/run_sonnet_costcheck.sh < sample.txt
f=$(realpath "$1"); d=$(dirname "$(dirname "$f")")/costcheck_sonnet; n=$(basename "${f%.txt}")
[ -s "$d/$n.json" ] && exit 0
cd "$(mktemp -d)" && claude -p "$(cat "$f")" --model claude-sonnet-5 --output-format json --restricted --strict-mcp-config \
  < /dev/null > "$d/$n.json" 2>&1
