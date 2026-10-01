#!/usr/bin/env bash
# Score one prompt file with Claude Haiku 4.5 (second LLM) in clean CLI mode; skips files already scored.
# Writes the raw JSON reply (with the model id) to haiku/<name>.json and the text reply to haiku/<name>.out.
# Usage (from results/frontier/): ls batches/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_haiku_batch.sh
f=$(realpath "$1"); d=$(dirname "$(dirname "$f")")/haiku; n=$(basename "${f%.txt}")
[ -s "$d/$n.out" ] && exit 0
cd "$(mktemp -d)" && claude -p "$(cat "$f")" --model claude-haiku-4-5-20251001 --output-format json --restricted \
  --strict-mcp-config < /dev/null > "$d/$n.json" 2>&1
python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['result'])" "$d/$n.json" > "$d/$n.out"
