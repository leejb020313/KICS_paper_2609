"""Check-worthiness scoring prompt, reproduced from NN-PPI (arXiv 2608.30731) Figure 2.

The paper's six few-shot examples (one per tier) are not printed in the paper
text -- only referenced as a {{examples}} placeholder -- so these are our own,
built to match the paper's stated tier definitions and target score ranges.
This is a known, unavoidable deviation from exact reproduction; note it in
writeup.
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
the claim check-worthiness criteria above and assign it a score of a value between 0 and 1
on how confident you are of it being a check-worthy claim. Assign it with higher confidence
score if you are more confident about the statement being a claim, else assign it lower
confidence score. Alongside claim check-worthiness confidence score, also try to provide a
justification for your confidence score. Provide the justification in natural language and
in no more than 100 words. ## Input statement:
{claim} ## Output format: Only output a JSON object with confidence_score and justification
that can be parsed by a JSON parser. Do not output any other text. Strictly format the
output as JSON object below. {{ "confidence_score": <a float value between 0 and 1>,
"justification": <a short natural language justification for the confidence_score> }}"""

# One example per tier, matching the paper's six-tier criteria list.
FEW_SHOT_EXAMPLES = [
    {
        "claim": "A new government-funded study found that the national unemployment rate fell by 2 percentage points over the last year.",
        "confidence_score": 0.95,
        "justification": "High-stakes, society-level, study-based quantitative claim about national economic performance.",
    },
    {
        "claim": "The new trade agreement will cut tariffs on manufactured goods across the entire sector by 15 percent.",
        "confidence_score": 0.8,
        "justification": "Broad, sector-wide policy mechanism with a specific quantitative claim.",
    },
    {
        "claim": "Crime went up in our county last year, and I think it's because of a few bad policy choices.",
        "confidence_score": 0.55,
        "justification": "Mid-tier, localized claim mixing a factual assertion with opinion.",
    },
    {
        "claim": "A man in my neighborhood said his power bill tripled last month.",
        "confidence_score": 0.3,
        "justification": "Isolated, hearsay anecdote with a loosely phrased generalization.",
    },
    {
        "claim": "I remember when I was a kid, my grandmother used to bake bread every Sunday morning.",
        "confidence_score": 0.1,
        "justification": "Personal, nostalgic story with no policy or society-level relevance.",
    },
    {
        "claim": "We have to fight for a brighter future for our children, and I believe we will win.",
        "confidence_score": 0.05,
        "justification": "Rhetorical, campaign-slogan-style statement with no verifiable factual content.",
    },
]


def build_examples_block() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(
            f'Statement: "{ex["claim"]}"\n'
            f'{{"confidence_score": {ex["confidence_score"]}, '
            f'"justification": "{ex["justification"]}"}}'
        )
    return "\n\n".join(lines)


def build_prompt(claim: str) -> str:
    return SYSTEM_PROMPT.format(examples=build_examples_block(), claim=claim)
