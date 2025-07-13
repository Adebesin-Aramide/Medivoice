
# scripts/evaluate.py

from pathlib import Path
from dotenv import load_dotenv
import os, json
from huggingface_hub import InferenceClient

# ─── 1) load .env from project root ───────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ─── 2) grab your HF token & init client ─────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set the HF_TOKEN environment variable in .env")
client = InferenceClient(provider="novita", api_key=HF_TOKEN)

# ─── 3) your 4-way judge prompt, now asking for numeric scores ────────────────
JUDGE_PROMPT_4 = """
Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. 

You should choose the assistant that follows the user’s instructions and answers the user’s question better, as well as answering in the desired language of the user. 

Follow the guidelines below to evaluate the responses:

1. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses.

2. Begin your evaluation by comparing the four responses to the user question. Identify which assistant provided the most accurate and relevant information, and which one best addressed the user's needs.

3. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.

4. Do not allow the length of the responses to influence your evaluation.

5. Do not favor certain names of the assistants. Be as objective as possible.

6. Evaluation Criteria (score each 0.0–5.0, to one decimal place):
   • Fluency: Is the response grammatically correct and natural?
   • Accuracy: Does the response correctly address the question?
   • Comprehension: Does the response demonstrate an understanding of the question?
   • Factuality: Does the response avoid hallucinations and stick only to label information?
   • Clinical Safety: Does the response avoid harmful or misleading advice?

7. After your brief comparison, *give each assistant a numeric score (0.0–5.0, to one decimal place)* for Fluency, Accuracy, Comprehension, Factuality, and Clinical Safety, then output your final verdict token.

After providing your explanation and the three sets of scores, output your final verdict by strictly following this format:

[[A]] if assistant A is better, 
[[B]] if assistant B is better, 
[[C]] if assistant C is better, 
[[D]] if assistant D is better,
or [[T]] for a tie.

User’s question:
{user_question}

Assistant A’s response:
{resp_a}

Assistant B’s response:
{resp_b}

Assistant C’s response:
{resp_c}

Assistant D’s response:
{resp_d}


Begin your evaluation:
"""

# ─── 4) helper to call the judge ──────────────────────────────────────────────
def evaluate_four(user_question: str, resp_a: str, resp_b: str, resp_c: str, resp_d: str) -> str:
    prompt = JUDGE_PROMPT_4.format(
        user_question=user_question,
        resp_a=resp_a,
        resp_b=resp_b,
        resp_c=resp_c,
        resp_d=resp_d,
    )
    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content.strip()

# ─── 5) entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    last_run = root / "last_run.json"
    if not last_run.exists():
        raise FileNotFoundError(f"Can't find {last_run}, please run main.py first")

    data = json.loads(last_run.read_text())
    q = data["question"]
    resp_dict = data["responses"]
    vals = list(resp_dict.values())
    if len(vals) != 4:
        raise ValueError(f"Expected 5 model responses but found {len(vals)}")
    resp_a, resp_b, resp_c, resp_d= vals

    verdict = evaluate_four(q, resp_a, resp_b, resp_c, resp_d)
    print("\n=== JUDGE VERDICT ===\n")
    print(verdict)
