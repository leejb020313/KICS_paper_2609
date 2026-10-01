#!/usr/bin/env bash
# Score one prompt file with Claude Sonnet 5 through the Claude Code CLI; skips files already scored.
# The model id is pinned: the September runs used claude-sonnet-5, and the CLI alias "sonnet" now resolves to a newer model.
# Usage (from results/llm_scores/): ls sonnet5/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_frontier_batch.sh
f=$1; o=${f%.txt}.out
[ -s "$o" ] && exit 0
claude -p "$(cat "$f")" --model claude-sonnet-5 --output-format text > "$o" 2>&1
