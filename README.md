# MediVoice

**MediVoice** is an end-to-end, modular Python project for capturing medication labels via webcam (or existing images), extracting text with OCR, processing user voice commands, and generating rich, context-aware responses via a hosted LLM (e.g., Mistral) on Hugging Face.

---

## 🚀 Project Overview

1. **Image Capture & OCR**

   * Automatically find and process the latest `label*.jpg` files in your configured `labels/` directory.
   * Uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) to extract raw text from medication labels.
   * Cleans and normalizes OCR output (lowercase, remove non-alphanumerics, drop stopwords).

2. **Speech-to-Text**

   * Scans your `Recordings/` directory for new audio files (`.wav`, `.mp3`, `.m4a`, etc.).
   * Converts to WAV (if needed) and transcribes using OpenAI Whisper or Google Speech API.

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
├── labels/             # Drop or capture label images here (label1.jpg, label2.jpg...)
├── Recordings/         # Place audio files (.wav, .mp3, .m4a) here
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

2. **Create & activate a virtual environment**

   ```bash
   python -m venv medivoiceenv
   source medivoiceenv/bin/activate   # macOS/Linux
   medivoiceenv\Scripts\activate.bat  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   * `MEDIVOICE_LABEL_DIR`: (optional) override default `labels/` path.
   * `MEDIVOICE_AUDIO_DIR`: (optional) override default `Recordings/` path.
   * `HUGGINGFACEHUB_API_TOKEN`: your Hugging Face token for the Inference API.

   Example (macOS/Linux):

   ```bash
   export MEDIVOICE_LABEL_DIR="/path/to/labels"
   export MEDIVOICE_AUDIO_DIR="/path/to/recordings"
   export HUGGINGFACEHUB_API_TOKEN="hf_xxx"
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

## 🤝 Contributing

1. Create a new feature branch: `git checkout -b feature/awesome-feature`
2. Write clear, well-documented code in `scripts/`.
3. Add unit tests (if applicable).
4. Submit a Pull Request describing your changes.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

> *Happy coding & stay healthy!*
