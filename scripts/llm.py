# scripts/llm.py

import os
from transformers import pipeline

# 1) instantiate the local Mistral-Instruct pipeline
#    * don't pass `device=` if you loaded with accelerate
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    # device_map="auto",      # optional if you want HF to shard across GPUs
    # the following defaults can be overridden per-call if you like:
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,    # <— strip off the prompt
)

def build_prompt(ocr_lines: list[str], user_question: str) -> str:
    """
    Combine OCR + user question into a single instruction,
    with explicit guidelines about style and content.
    """
    ocr_block = "\n".join(f"- {line}" for line in ocr_lines)

    return (
        "You are a knowledgeable, friendly medical assistant.\n"
        "Your task is to answer the user’s question based on the provided medication label.\n\n"

        "Please follow these guidelines when answering:\n"
        "  • Be clear and concise.\n"
        "  • Be precise—stick exactly to what you know from the label.\n"
        "  • Use simple, everyday English.\n"
        "  • Do NOT invent or hallucinate any details. "
        "If the label doesn’t provide enough information, "
        'respond with "visit the nearest hospital for more details".\n\n'
        "Medication label text:\n"
        f"{ocr_block}\n\n"
        "User’s question:\n"
        f"{user_question}\n\n"
        "Answer:"
    )


def generate_answer(ocr_lines: list[str], user_question: str) -> str:
    prompt = build_prompt(ocr_lines, user_question)

    # 2) run the model — prompt won't be echoed
    outputs = generator(
        prompt,
        # you can still tweak per-call if needed:
        # max_new_tokens=256,
        # do_sample=True,
        # temperature=0.5,
        # return_full_text=False
    )

    # 3) pipeline returns List[dict]
    answer = outputs[0]["generated_text"]
    return answer.strip()
