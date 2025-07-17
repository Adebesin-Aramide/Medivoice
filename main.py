import os
import json
from config import LABEL_DIR, AUDIO_DIR
from scripts.label_extraction import get_latest_image, extract_text, preprocess
from scripts.speech_to_text  import speech_to_text
from scripts.llm import generate_all_answers, MODEL_PROVIDER

def main():
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 1) OCR
    img   = get_latest_image(LABEL_DIR)
    raw   = extract_text(img)
    clean = preprocess(raw)

    print("=== OCR TEXT ===")
    for l in raw:
        print(" •", l)

    # 2) STT
    audio_files = sorted(
        [f for f in os.listdir(AUDIO_DIR)
         if f.lower().endswith((".wav", ".mp3", ".m4a", ".aac"))],
        key=lambda fn: os.path.getctime(os.path.join(AUDIO_DIR, fn))
    )
    if not audio_files:
        print("No audio files in", AUDIO_DIR)
        return

    user_command = speech_to_text(os.path.join(AUDIO_DIR, audio_files[-1]))
    print("\n=== USER QUESTION ===\n", user_command)

    # 3) Call all models
    print("\n=== MODEL RESPONSES ===")
    all_answers = generate_all_answers(clean, user_command)
    for model_id, ans in all_answers.items():
        print(f"\n--- {model_id} ---\n{ans}")

    # 4) Save for evaluation
    with open("last_run.json", "w") as f:
        json.dump({
            "question":  user_command,
            "raw_ocr":   raw,
            "clean_ocr": clean,
            "responses": all_answers
        }, f, indent=2)

    print("✅  Saved to last_run.json — ready for `evaluate.py`.")


    # 5) Select the best model:
    best_model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    answer = all_answers[best_model_id]

    print(f"\n🏆 Using {best_model_id}:\n{answer}")
 

if __name__ == "__main__":
    main()
