import os
import time
from huggingface_hub import InferenceClient
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set the HF_TOKEN environment variable")

# Configure with timeout and retries
CLIENTS = {
    "novita": InferenceClient(
        provider="novita", 
        api_key=HF_TOKEN,
        timeout=30
    ),
    "featherless-ai": InferenceClient(
        provider="featherless-ai", 
        api_key=HF_TOKEN,
        timeout=30
    ),
    "together": InferenceClient(
        provider="together", 
        api_key=HF_TOKEN,
        timeout=30
    ),
}

MODEL_PROVIDER = {
    "mistralai/Mistral-7B-Instruct-v0.3": "novita",
    "meta-llama/Meta-Llama-3-8B-Instruct": "novita",
    "Qwen/Qwen2.5-7B-Instruct": "featherless-ai",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "together",
}

def build_prompt(ocr_lines: List[str], user_question: str) -> str:
    # Turn the OCR lines into a bullet list
    ocr_block = "\n".join(f"- {line}" for line in ocr_lines)

    return (
        "<s>[INST]\n"
        "You are a medication safety assistant helping visually impaired users, your sole source of information is the medication label.\n\n"

        "Follow these rules strictly (25-word maximum):\n"
        "  1. ANSWER ONLY USING THE LABEL TEXT - do NOT infer or add anything extra.\n"
        "  2. Keep it under 25 words, in plain, everyday English.\n"
        "  3. If the label lacks the answer, respond: “Consult a doctor.”\n\n"

        "MEDICATION LABEL:\n"
        f"{ocr_block}\n\n"

        "USER QUESTION:\n"
        f"{user_question}\n\n"

        "YOUR ANSWER:[/INST]"
    )


def generate_all_answers(ocr_lines: List[str], user_question: str) -> Dict[str, str]:
    prompt = build_prompt(ocr_lines, user_question)
    answers = {}
    
    for model_id, provider in MODEL_PROVIDER.items():
        try:
            print(f"\nCalling {model_id}...")
            start_time = time.time()
            
            client = CLIENTS[provider]
            completion = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            
            if completion and completion.choices:
                answer = completion.choices[0].message.content.strip()
                answers[model_id] = answer
                print(f"Response received in {time.time()-start_time:.1f}s")
            else:
                answers[model_id] = "Error: Empty response"
                
        except Exception as e:
            answers[model_id] = f"Error: {str(e)}"
            print(f"Failed on {model_id}: {e}")
    
    return answers