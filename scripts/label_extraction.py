import os
import glob
import re
import easyocr

# your existing stop-word set
_STOP_WORDS = {
    'a','an','and','are','as','at','be','but','by','for','if','in','into','is',
    'it','no','not','of','on','or','s','such','t','that','the','their','then',
    'there','these','they','this','to','was','will','with'
}

def get_latest_image(image_dir: str, prefix: str = "label", ext: str = ".jpg") -> str:
    """
    Scan `image_dir` for files named like prefixN.ext and
    return the most recently created one.
    """
    pattern = os.path.join(image_dir, f"{prefix}*{ext}")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern}")
    return max(candidates, key=os.path.getctime)

def extract_text(image_path: str, lang: list = ['en'], gpu: bool = True) -> list[str]:
    """
    Run EasyOCR and return raw lines of text.
    """
    reader = easyocr.Reader(lang, gpu=gpu)
    return reader.readtext(image_path, detail=0, paragraph=True)

def clean_text(line: str) -> str:
    """
    Basic cleanup: lowercase, strip punctuation, remove stop-words, etc.
    """
    text = re.sub(r'[^a-z0-9\s]', ' ', line.lower())
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS]
    return " ".join(tokens)

def preprocess(lines: list[str]) -> list[str]:
    """
    Apply clean_text() to each OCR line.
    """
    return [clean_text(l) for l in lines]
