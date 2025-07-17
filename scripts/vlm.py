# scripts/vlm.py

import os
import time
import json
import base64
import warnings
from pathlib import Path
from huggingface_hub import InferenceClient

# ─── 1) Configuration ─────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set the HF_TOKEN environment variable (your HF API key)")

# Choose the provider & model for your vision‐language chat
CLIENT = InferenceClient(provider="nebius", api_key=HF_TOKEN, timeout=30)
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

SYSTEM_INSTRUCTIONS = (
    "You are a friendly, patient, and highly accurate medical assistant. "
    "Your task is to answer the user’s question strictly based on the contents of the provided medication label image. "
    "Follow these rules:\n"
    "  • Only use information actually visible on the label—do NOT hallucinate or invent details.\n"
    "  • Be clear and concise, in simple everyday language.\n"
    "  • If the label does not show the answer, reply: "
    "“I’m not sure from this label alone; please check with a pharmacist or doctor.”\n"
    "  • Provide dosage, usage, safety, or storage details exactly as written.\n"
    "  • Keep your answer polite and patient‑focused."
)

# ─── 2) Helpers ────────────────────────────────────────────────────────────────
def _image_to_data_url(image_path: str) -> str:
    """Embed a local JPG/PNG into a base64 data URL for the API."""
    with open(image_path, "rb") as img_f:
        b64 = base64.b64encode(img_f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lstrip(".").lower()
    return f"data:image/{ext};base64,{b64}"

# ─── 3) Core V‑L function ─────────────────────────────────────────────────────
def generate_answer_via_chat(image_path: str, user_question: str) -> str:
    """
    Sends a vision+text prompt to the V‑L chat model and returns its reply.

    Args:
        image_path: local path to the medication‐label image
        user_question: the question string to ask about that label

    Returns:
        The model’s plaintext answer.
    """
    # 1) Encode the image
    image_data = _image_to_data_url(image_path)

    # 2) Build the two messages
    system_msg = {"role": "system", "content": SYSTEM_INSTRUCTIONS}
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Question about this medication label: {user_question}"},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
    }

    # 3) Call the API
    start = time.time()
    completion = CLIENT.chat.completions.create(
        model=MODEL_ID,
        messages=[system_msg, user_msg],
        max_tokens=256,
        temperature=0.0  # deterministic
    )
    elapsed = time.time() - start

    # 4) Extract & return
    if not completion.choices:
        raise RuntimeError("Empty response from V‑L model")
    answer = completion.choices[0].message.content.strip()

    # (optional) print timing
    print(f"🖼→💬 V‑L call done in {elapsed:.1f}s")
    return answer
