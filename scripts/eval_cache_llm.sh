#!/usr/bin/env bash
# Cache the 100-run predictions of one LLM score folder (results/llm_scores/<folder>) for both datasets.
#   bash scripts/eval_cache_llm.sh sonnet5_cli_default
cd "$(dirname "$0")/.." || exit 1
export CWC_CACHE="$PWD/results/cache" CWC_LLM="$1"
mkdir -p results/logs/run100
for ds in clef cb; do
  uv run --locked python -c "import sys; sys.path.insert(0, 'scripts'); import final_eval as fe; fe.evaluate('$ds', True)" \
    > "results/logs/run100/eval_${1}_${ds}.log" 2>&1 &
done
wait
ls results/cache
