#!/usr/bin/env bash
# Score one prompt file with Claude Sonnet 5 through the Claude Code CLI; skips files already scored.
# Usage (from results/frontier/): ls batches/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_frontier_batch.sh
# --restricted / --strict-mcp-config and a neutral working directory keep the local Claude Code setup
# (output styles, plugin hooks, project instructions) out of the call, so the reply is the bare JSON
# object, as in the original September runs (added for the CT22 batches; a clean re-score of cb_0000
# matched the original: AUC 0.948 vs 0.953).
f=$(realpath "$1"); o=${f%.txt}.out
[ -s "$o" ] && exit 0
cd "$(mktemp -d)" && claude -p "$(cat "$f")" --model sonnet --output-format text --restricted --strict-mcp-config \
  < /dev/null > "$o" 2>&1
