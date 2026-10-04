#!/usr/bin/env bash
# Option 1 / option 2 for the three LLM score sets in parallel; pass --knn for option 2.
cd "$(dirname "$0")/.." || exit 1
tag=$([ "$1" = "--knn" ] && echo knn || echo op)
for llm in sonnet5 sonnet5_cli_default haiku45; do
  CWC_LLMS=$llm uv run --locked python scripts/operating_point.py $1 > "results/logs/${tag}_${llm}.log" 2>&1 &
done
wait
echo "done $tag"
