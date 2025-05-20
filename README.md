# MediVoice

**MediVoice** is a voice-based mobile app that helps visually impaired people manage their medications on their own. With this app, users can take a picture of a drug label and ask questions by speaking. The app then reads the label, understands the question, and gives a clear voice response. It tells users things like the name of the medicine 


---

## 🚀 Project Overview

1. **Image Capture & OCR**

   * Automatically find and process the latest image files in the labels directory.
   * Uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) to extract raw text from medication labels.
   * Cleans and normalizes OCR output (lowercase, remove non-alphanumerics, drop stopwords).

2. **Speech-to-Text**

   * Scans your `Recordings/` directory for new audio files (`.wav`, `.mp3`, `.m4a`, etc.).
   * Converts to WAV (if needed) and transcribes using OpenAI Whisper.

3. **LLM Inference**

   * Combines clean OCR text and user question into a natural prompt.
   * Sends the prompt to a hosted Hugging Face Inference endpoint (no local model download).
   * Returns a clear, conversational answer or a polite fallback if data is insufficient.

4. **Workflow Automation**

   * `main.py` ties together all pipelines, making it easy to run everything in one command.

---

## 📂 Repository Structure

```
MediVoice/
├── README.md           # This file
├── main.py             # Entry point: runs OCR, STT, and LLM inference
├── config.py           # Directory paths & environment overrides
├── requirements.txt    # Python dependencies
├── .env                # Store your API keys
└── scripts/
    ├── label_extraction.py  # OCR helpers: get_latest_image, extract_text, preprocess
    ├── speech_to_text.py    # Audio helpers: convert_to_wav, speech_to_text
    └── llm.py               # Inference helpers: build_prompt & generate_answer
```

---

## 🔧 Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourorg/MediVoice.git
   cd MediVoice
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the full pipeline with:

```bash
python main.py
```

* **OCR** will process the newest `label*.jpg` in `labels/`.
* **Speech-to-Text** will transcribe the newest audio file in `Recordings/`.
* **LLM Inference** will generate a response using your Hugging Face hosted model.


---
