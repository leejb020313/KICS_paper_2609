"""De-quantize the check-worthiness score by reading token logprobs.

WHY (measured, see ../prompt-reorder-ablation/channel_capacity.py):
    Gemma 3 4B collapses every claim into ~13 round-number score values. On
    CLEF the oracle ceiling of that channel (best possible mapping of those 13
    values to labels) is 0.780 -- exactly equal to what a plain 0.5 threshold
    already achieves. There is literally zero headroom left for ANY post-hoc
    calibration on the verbalized score. On ClaimBuster, 372 claims all land on
    "0.70" and P(gold=1)=0.500 inside that bucket: a coin flip that no
    downstream method can resolve, because the inputs are identical symbols.
    Meanwhile Claude Sonnet 5, on the same 50 claims, spreads them over 18
    values and carries 0.945 bits of the 1.0-bit label entropy (Gemma: 0.484).

    The quantization happens when a token is SAMPLED. The model's internal
    state is continuous. Reading the next-token distribution recovers what the
    sampler threw away -- at zero extra compute, since those logits are
    computed anyway.

THREE SCORING MODES (all single forward pass, all continuous output):

  yesno   P(check-worthy) = P("Yes") / (P("Yes") + P("No")) from the first
          generated token's logprobs. Fully bypasses verbalized confidence.
          The cleanest probe: no digits, no JSON, no rounding.

  digit   Ask for a single digit 0-9, then take the expectation over the whole
          digit distribution: E[score] = sum_d P(d) * d/9. Keeps the ordinal
          10-point scale of the original task but returns a continuous value.

  verbal  Original NN-PPI prompt/JSON, but instead of parsing the emitted
          number we locate the score's first decimal-digit token and take its
          expected value. Closest to the paper, hardest to parse robustly.

Usage (on the GPU box running llama-server):
    python score_with_logprobs.py --mode yesno --dataset ../../data/processed/clef_test.csv \
        --out results/clef_test_yesno.jsonl
    python score_with_logprobs.py --mode digit --dataset ../../data/processed/clef_test.csv \
        --out results/clef_test_digit.jsonl

Server must expose logprobs. llama-server does on /v1/chat/completions:
    llama-server -m <gemma3-4b>.gguf --port 8080 -c 8192 -ngl 99
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from nnppi.prompt import build_examples_block

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"

CRITERIA = """# YOUR ROLE
You are an impartial fact-checker. You are aware of what kind of statements that goes
around news and published media are fact-check-worthy claims or not based on following
check-worthiness criteria.
• High-stakes, society-level, quantitative or study-based claims are very highly
check-worthy.
• Broad policy mechanism or sector-wide quantitative claims are highly check-worthy.
• Mid-tier, localized or mixed claims with numbers/opinions are medium check-worthy.
• Isolated incidents, hearsay, or loosely phrased generalizations are low check-worthy.
• Personal stories, greetings, meta, nostalgia, logistics are very low check-worthy.
• Statements containing factual claims that are not check-worthy by above check-worthiness
criteria or non claim statements (e.g. opinions, speculations, feelings, rhetorical
statements, campaign slogans, predictions) are not check-worthy.
## Some examples for claim check-worthiness are below. {examples}
"""

YESNO_TASK = """# YOUR TASK
Statement: {claim}

Is this statement a check-worthy claim by the criteria above?
Answer with exactly one word: Yes or No.
Answer:"""

DIGIT_TASK = """# YOUR TASK
Statement: {claim}

Rate how check-worthy this statement is by the criteria above, on a scale where
0 means definitely not check-worthy and 9 means definitely check-worthy.
Answer with exactly one digit from 0 to 9, nothing else.
Answer:"""

VERBAL_TASK = """# YOUR TASK
You will be provided a statement which can be a claim or not. Make your best judgement using
the claim check-worthiness criteria above and assign it a score of a value between 0 and 1
on how confident you are of it being a check-worthy claim. ## Input statement:
{claim} ## Output format: Output only the score as a decimal number between 0 and 1.
Score: 0."""


def build_prompt(mode: str, claim: str) -> str:
    head = CRITERIA.format(examples=build_examples_block())
    task = {"yesno": YESNO_TASK, "digit": DIGIT_TASK, "verbal": VERBAL_TASK}[mode]
    return head + task.format(claim=claim)


def call_llm(base_url, prompt, top_logprobs, max_tokens, temperature, model=None, timeout=600):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        # Gemma has no thinking mode, but be explicit in case the template honors it:
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        payload["model"] = model  # required by Ollama, ignored by llama-server
    r = requests.post(base_url, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return body["choices"][0]


def token_distribution(choice, position=0):
    """Return [(token_text, probability)] for the generated token at `position`."""
    content = (choice.get("logprobs") or {}).get("content") or []
    if position >= len(content):
        return []
    entry = content[position]
    tops = entry.get("top_logprobs") or []
    return [(t["token"], math.exp(t["logprob"])) for t in tops]


def first_informative_position(choice, predicate, max_scan=8):
    """Find the first generated token whose top-alternatives satisfy `predicate`.
    Guards against the model opening with whitespace/markdown/a thinking preamble
    before answering."""
    content = (choice.get("logprobs") or {}).get("content") or []
    for pos in range(min(len(content), max_scan)):
        dist = token_distribution(choice, pos)
        if predicate(dist):
            return pos, dist
    return None, []


# A stray "Yes" sitting at p=0.002 inside a thinking preamble ("Okay, ...") must
# NOT be mistaken for the answer position -- verified against Ollama's top-10 on
# qwen3:0.6b, where exactly that happens. Require the target tokens to actually
# own the position before reading a score off it.
MIN_ANSWER_MASS = 0.5


def _ascii_digit(tok):
    """int() on a token that merely passes str.isdigit() crashes: isdigit() is True
    for superscripts and other Unicode digits, and llama/Ollama can hand back
    mojibake bytes. Seen live as ValueError: invalid literal for int() '\\ufffd'.
    Only accept a bare ASCII 0-9."""
    s = tok.strip()
    return int(s) if len(s) == 1 and "0" <= s <= "9" else None


def _leading_ascii_digit(tok):
    s = tok.strip()
    return int(s[0]) if s and "0" <= s[0] <= "9" else None


def score_yesno(choice, min_mass=MIN_ANSWER_MASS):
    """P(check-worthy) = P(Yes) / (P(Yes) + P(No)), normalized over the two poles."""
    def has_polarity(dist):
        mass = sum(p for t, p in dist if t.strip().lower().startswith(("yes", "no")))
        return mass >= min_mass

    pos, dist = first_informative_position(choice, has_polarity)
    if pos is None:
        return None, None
    p_yes = sum(p for t, p in dist if t.strip().lower().startswith("yes"))
    p_no = sum(p for t, p in dist if t.strip().lower().startswith("no"))
    total = p_yes + p_no
    if total <= 0:
        return None, None
    p = p_yes / total
    # Gemma 3 4B saturates hard here: measured p_yes values were 1.8e-05, 2.1e-05,
    # 1.6e-05 for three different non-check-worthy claims, and 0.9999998 /
    # 1.0000000 for two check-worthy ones. As a probability that is a 2-value
    # channel -- WORSE than the 13-value verbalized one. But those tail values are
    # distinct and ordered, so the information survives in log-odds. Record the
    # logit; analyze_channel.py should evaluate on it, not on p.
    eps = 1e-12
    logit = math.log(max(p, eps) / max(1.0 - p, eps))
    return p, {"p_yes": p_yes, "p_no": p_no, "mass_on_poles": total, "logit": logit}


def score_digit(choice, min_mass=MIN_ANSWER_MASS):
    """E[score] over the full 0-9 digit distribution, rescaled to [0, 1]."""
    def has_digit(dist):
        mass = sum(p for t, p in dist if _ascii_digit(t) is not None)
        return mass >= min_mass

    pos, dist = first_informative_position(choice, has_digit)
    if pos is None:
        return None, None
    weights = {}
    for t, p in dist:
        d = _ascii_digit(t)
        if d is not None:
            weights[d] = weights.get(d, 0.0) + p
    total = sum(weights.values())
    if total <= 0:
        return None, None
    expected = sum(d * p for d, p in weights.items()) / total
    return expected / 9.0, {"digit_probs": {str(k): round(v / total, 5) for k, v in sorted(weights.items())},
                            "mass_on_digits": total}


def score_verbal(choice, min_mass=MIN_ANSWER_MASS):
    """Prompt is primed with 'Score: 0.' so the first token is the tenths digit.
    E[tenths] / 10 gives a continuous score in [0, 1]."""
    def has_digit(dist):
        mass = sum(p for t, p in dist if _leading_ascii_digit(t) is not None)
        return mass >= min_mass

    pos, dist = first_informative_position(choice, has_digit)
    if pos is None:
        return None, None
    weights = {}
    for t, p in dist:
        d = _leading_ascii_digit(t)
        if d is not None:
            weights[d] = weights.get(d, 0.0) + p
    total = sum(weights.values())
    if total <= 0:
        return None, None
    expected = sum(d * p for d, p in weights.items()) / total
    return expected / 10.0, {"tenths_probs": {str(k): round(v / total, 5) for k, v in sorted(weights.items())},
                             "mass_on_digits": total}


SCORERS = {"yesno": score_yesno, "digit": score_digit, "verbal": score_verbal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SCORERS), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=None,
                    help="required by Ollama (e.g. gemma3:4b); ignored by llama-server")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="we only need the first answer token, but leave room for a "
                         "short preamble that first_informative_position can skip")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="irrelevant to the logprob distribution itself, but keeps the "
                         "sampled token deterministic for auditing")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["Sentence_id"])
        print(f"Resuming: {len(done_ids)} already scored.")

    remaining = df[~df["Sentence_id"].isin(done_ids)]
    print(f"{len(remaining)} claims to score out of {len(df)} (mode={args.mode}).")

    scorer = SCORERS[args.mode]
    n_failed = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for n_done, (_, row) in enumerate(remaining.iterrows(), 1):
            prompt = build_prompt(args.mode, row["text"])
            score, detail, sampled = None, None, None
            for attempt in range(args.retries):
                try:
                    choice = call_llm(args.base_url, prompt, args.top_logprobs,
                                      args.max_tokens, args.temperature, model=args.model)
                    sampled = choice.get("message", {}).get("content")
                    score, detail = scorer(choice)
                    break
                except Exception as e:
                    print(f"  [{row['Sentence_id']}] attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            if score is None:
                n_failed += 1

            f.write(json.dumps({
                "Sentence_id": row["Sentence_id"],
                "text": row["text"],
                "label": int(row["label"]),
                "confidence_score": score,
                "parse_ok": score is not None,
                "detail": detail,
                "sampled_text": sampled,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(remaining)} done ({n_failed} failed)")

    print(f"Done. {n_failed} failures.")


if __name__ == "__main__":
    main()
