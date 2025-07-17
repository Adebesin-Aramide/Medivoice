# main.py

import os
import json
from pathlib import Path

from config import LABEL_DIR, AUDIO_DIR
from scripts.label_extraction import get_latest_image
from scripts.speech_to_text import speech_to_text
from scripts.vlm import generate_answer_via_chat

def main():
    # Ensure our working folders exist
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 1) Pick the latest label image via your helper
    img_path = get_latest_image(str(LABEL_DIR))
    print(f"🏷️  Label image path: {img_path}")

    # 2) Pick the latest audio recording
    audio_candidates = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".aac"))
    ]
    if not audio_candidates:
        print("No audio files in", AUDIO_DIR)
        return
    # Find newest by creation time
    latest_audio = max(
        audio_candidates,
        key=lambda fn: os.path.getctime(os.path.join(AUDIO_DIR, fn))
    )
    audio_path = os.path.join(AUDIO_DIR, latest_audio)
    print(f"🎙  Audio path: {audio_path}")

    # 3) Transcribe user’s question
    user_question = speech_to_text(audio_path)
    print("\n=== USER QUESTION ===")
    print(user_question)

    # 4) Call V‑L model
    print("\n🤖  Generating answer via V‑L model…")
    answer = generate_answer_via_chat(img_path, user_question)

    print("\n🔍 Model’s Answer:")
    print(answer)

    # 5) Save for later (evaluation / TTS etc.)
    out = {
        "question": user_question,
        "image_path": img_path,
        "audio_path": audio_path,
        "answer": answer
    }
    with open("vlm_run.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n✅  Saved to vlm_run.json")

if __name__ == "__main__":
    main()
