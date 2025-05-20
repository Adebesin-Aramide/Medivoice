# main.py

import os
from config import LABEL_DIR, AUDIO_DIR
from scripts.label_extraction import (
    get_latest_image, extract_text, preprocess
)
from scripts.speech_to_text import convert_to_wav, speech_to_text
from scripts.llm import generate_answer  # <-- import our new function

def main():
    # 1) Ensure our working folders exist
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 2) OCR pipeline
    print("🖼  Running OCR pipeline…")
    img_path       = get_latest_image(LABEL_DIR)
    raw_lines      = extract_text(img_path)
    clean_lines    = preprocess(raw_lines)

    print("\n=== RAW OCR TEXT ===")
    for line in raw_lines:
        print(" •", line)
    print("\n=== CLEANED OCR TEXT ===")
    for line in clean_lines:
        print(" •", line)

    # 3) Speech-to-text pipeline
    print("\n🎙  Running Speech-to-Text pipeline…")
    audio_files = sorted(
        [
            f for f in os.listdir(AUDIO_DIR)
            if f.lower().endswith((".wav", ".mp3", ".m4a", ".aac"))
        ],
        key=lambda fn: os.path.getctime(os.path.join(AUDIO_DIR, fn))
    )
    if not audio_files:
        print("No audio files found in", AUDIO_DIR)
        return

    latest_audio = os.path.join(AUDIO_DIR, audio_files[-1])
    wav_path     = os.path.join(AUDIO_DIR, "converted.wav")
    convert_to_wav(latest_audio, wav_path)
    user_command = speech_to_text(wav_path)

    print("\n=== USER COMMAND ===")
    print(user_command)

    # 4) LLM inference
    print("\n🤖  Generating answer from Mistral…")
    answer = generate_answer(clean_lines, user_command)

    print("\n🔍 Model’s Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
