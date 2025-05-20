import os
import speech_recognition as sr
from pydub import AudioSegment

def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    Convert any audio format (mp3, m4a, etc.) to WAV.
    """
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path

def speech_to_text(audio_path: str) -> str:
    """
    Use Google’s Speech API via speech_recognition.
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as src:
        audio_data = r.record(src)
    try:
        return r.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "[unintelligible audio]"
    except sr.RequestError as e:
        return f"[speech API error: {e}]"
