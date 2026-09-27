"""Write the batched frontier-LLM prompts (40 statements per file) into results/frontier/batches/.

The prompt keeps NN-PPI's check-worthiness criteria (cwcascade/prompt.py) and replaces the
single-statement task with a numbered list scored in one JSON object. Ids are positions in
data/frontier/{eval,calib}_set.json. Score the files with scripts/run_frontier_batch.sh.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import DATA, RESULTS, read_json  # noqa: E402
from cwcascade.prompt import SYSTEM_PROMPT  # noqa: E402

BATCH = 40
CRITERIA = SYSTEM_PROMPT.split("## Some examples")[0]
TASK = ("\n# YOUR TASK\nBelow are numbered statements (id<TAB>statement). For EACH, assign a score between 0 and 1 "
        "for how confident you are that it is a check-worthy claim under the criteria above. Output ONLY a JSON "
        "object mapping id (as string) to score, e.g. {\"0\":0.9,\"1\":0.1}. Include every id.\n\n")


def main(out_dir=os.path.join(RESULTS, "frontier", "batches")):
    os.makedirs(out_dir, exist_ok=True)
    sets = {**read_json(os.path.join(DATA, "frontier", "eval_set.json")),
            **read_json(os.path.join(DATA, "frontier", "calib_set.json"))}
    for key, items in sets.items():
        for start in range(0, len(items), BATCH):
            lines = [f"{i}\t{items[i]['text']}" for i in range(start, min(start + BATCH, len(items)))]
            with open(os.path.join(out_dir, f"{key}_{start:04d}.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write(CRITERIA + TASK + "\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
