#!/usr/bin/env bash
# Score one prompt file with Claude Sonnet 5 in clean CLI mode, the same command as scripts/run_haiku_batch.sh
# (empty temp dir, --restricted ignores user/project settings). Skips files already scored.
# Writes the raw JSON reply (with the model id) to sonnet5/<name>.json and the text reply to sonnet5/<name>.out.
# Usage (from results/llm_scores/): ls sonnet5/*.txt | xargs -P 6 -n 1 bash ../../scripts/run_sonnet_clean_batch.sh
f=$(realpath "$1"); d=$(dirname "$f"); n=$(basename "${f%.txt}")
[ -s "$d/$n.out" ] && exit 0
cd "$(mktemp -d)" && claude -p "$(cat "$f")" --model claude-sonnet-5 --output-format json --restricted \
  --strict-mcp-config < /dev/null > "$d/$n.json" 2>&1
python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['result'])" "$d/$n.json" > "$d/$n.out"
# a usage-limit reply is not a score: remove it so the batch is re-run later
grep -q "hit your session limit\|usage limit" "$d/$n.json" && { rm -f "$d/$n.json" "$d/$n.out"; echo "LIMIT $n"; exit 1; }
exit 0
