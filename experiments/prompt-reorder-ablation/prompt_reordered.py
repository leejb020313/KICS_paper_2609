"""Check-worthiness scoring prompt -- REORDERED ablation of NN-PPI (arXiv 2608.30731) Figure 2.

Hypothesis: the original prompt's JSON schema asks for `confidence_score`
BEFORE `justification`. Under autoregressive generation this means the model
commits to a number before it has articulated any reasoning about the claim
-- the justification can only rationalize a score already fixed, never
inform it. We verified this empirically: every raw_response in
results/clef_test_scores.jsonl (main nnppi-reproduction branch) emits
confidence_score first, justification second.

This module is byte-for-byte identical to ../../src/nnppi/prompt.py except
for ONE change: the output schema and all six few-shot examples are
reordered so `justification` is generated first and `confidence_score`
second. If a model's score is meant to reflect the stated criteria, it
should now actually be able to condition on its own reasoning. This is the
same fix-shape as the "self-judgment reinsertion" prescription in the
author's own prior work (TicToc rule-baseline paper, KICS submission),
which recovered 72% of a knowing-doing gap by exposing the model's own
prior judgment before requiring the action/output that should follow from
it -- here the exposure is even cheaper: no extra turn, just a field-order
swap in the same generation.

Nothing else changes: same criteria text, same six claims/scores/
justifications, same {claim} slot, same "only output JSON" instruction.
This keeps the ablation clean -- one variable (field order) isolated.
"""

SYSTEM_PROMPT = """# YOUR ROLE
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
# YOUR TASK
You will be provided a statement which can be a claim or not. Make your best judgement using
the claim check-worthiness criteria above. First write a short justification (no more than
100 words) in natural language, reasoning step by step about which criteria above apply to
this statement. THEN, based on that justification, assign a confidence_score between 0 and 1
for how confident you are that it is a check-worthy claim -- higher if more confident it is a
check-worthy claim, lower otherwise. The confidence_score you output must be consistent with
the justification you just wrote. ## Input statement:
{claim} ## Output format: Only output a JSON object with justification and confidence_score
that can be parsed by a JSON parser. Do not output any other text. Strictly format the
output as JSON object below, in this exact key order. {{ "justification": <a short natural
language justification, reasoned before the score>, "confidence_score": <a float value
between 0 and 1, consistent with the justification above> }}"""

# Same six tier-examples as prompt.py, field order swapped to match the new schema.
FEW_SHOT_EXAMPLES = [
    {
        "claim": "A new government-funded study found that the national unemployment rate fell by 2 percentage points over the last year.",
        "justification": "High-stakes, society-level, study-based quantitative claim about national economic performance.",
        "confidence_score": 0.95,
    },
    {
        "claim": "The new trade agreement will cut tariffs on manufactured goods across the entire sector by 15 percent.",
        "justification": "Broad, sector-wide policy mechanism with a specific quantitative claim.",
        "confidence_score": 0.8,
    },
    {
        "claim": "Crime went up in our county last year, and I think it's because of a few bad policy choices.",
        "justification": "Mid-tier, localized claim mixing a factual assertion with opinion.",
        "confidence_score": 0.55,
    },
    {
        "claim": "A man in my neighborhood said his power bill tripled last month.",
        "justification": "Isolated, hearsay anecdote with a loosely phrased generalization.",
        "confidence_score": 0.3,
    },
    {
        "claim": "I remember when I was a kid, my grandmother used to bake bread every Sunday morning.",
        "justification": "Personal, nostalgic story with no policy or society-level relevance.",
        "confidence_score": 0.1,
    },
    {
        "claim": "We have to fight for a brighter future for our children, and I believe we will win.",
        "justification": "Rhetorical, campaign-slogan-style statement with no verifiable factual content.",
        "confidence_score": 0.05,
    },
]


def build_examples_block() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(
            f'Statement: "{ex["claim"]}"\n'
            f'{{"justification": "{ex["justification"]}", '
            f'"confidence_score": {ex["confidence_score"]}}}'
        )
    return "\n\n".join(lines)


def build_prompt(claim: str) -> str:
    return SYSTEM_PROMPT.format(examples=build_examples_block(), claim=claim)
