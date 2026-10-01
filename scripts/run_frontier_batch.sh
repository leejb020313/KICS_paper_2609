#!/usr/bin/env bash
# Score one prompt file with Claude Sonnet 5 through the Claude Code CLI; skips files already scored.
# Usage (from results/llm_scores/): ls sonnet5/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_frontier_batch.sh
f=$1; o=${f%.txt}.out
[ -s "$o" ] && exit 0
claude -p "$(cat "$f")" --model sonnet --output-format text > "$o" 2>&1
