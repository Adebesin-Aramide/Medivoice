from dotenv import load_dotenv
load_dotenv()        # <-- reads .env into os.environ

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def make_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

LABEL_DIR = make_dir(Path(os.getenv("MEDIVOICE_LABEL_DIR", BASE_DIR / "labels")))
AUDIO_DIR = make_dir(Path(os.getenv("MEDIVOICE_AUDIO_DIR", BASE_DIR / "Recordings")))
