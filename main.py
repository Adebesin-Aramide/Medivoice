# Ocr reader 
LABEL_DIR = '/content/drive/MyDrive/Medivoice/labels'
ocr_result = extract_text_from_latest_label(LABEL_DIR)

# -- Run Speech-to-Text --
input_path = "/content/drive/MyDrive/RECORD.m4a"
IPyAudio(input_path)  # Optional: plays back the audio in notebook

wav_path = convert_to_wav(input_path)
transcribed_text = speech_to_text(wav_path)
print("\n=== TRANSCRIPTION ===\n", transcribed_text)
