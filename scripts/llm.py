import os
from huggingface_hub import InferenceClient

def build_prompt(ocr_lines: list[str], user_question: str) -> str:
    ocr_block = "\n".join(f"- {line}" for line in ocr_lines)
    return f"""<|system|>
You are a knowledgeable, friendly medical assistant. Below is the raw text from a medication label and a user’s follow-up question.

Medication label text:
{ocr_block}

User’s question:
{user_question}

Please answer the user as clearly and naturally as you can, drawing only on the information in the label. If the label doesn’t provide enough detail to answer, say so in a polite way (for example, “I’m not sure from this label alone; you may need to check with a pharmacist or doctor.”).  
</s>
<|assistant|>
"""

def generate_answer(ocr_lines: list[str], user_question: str) -> str:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN in environment!")

    client = InferenceClient(token=token)
    prompt = build_prompt(ocr_lines, user_question)

    response = client.text_generation(
        model="HuggingFaceH4/zephyr-7b-beta",
        prompt=prompt,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    return response.strip()