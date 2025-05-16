import os
from pydub import AudioSegment
import whisper
from IPython.display import Audio

def convert_to_wav(input_path, output_path="converted.wav"):
    """
    Converts an audio file to WAV format using pydub.
    Supported: m4a, mp3, etc.
    """
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path


def speech_to_text(audio_path, model_size="base"):
    """
    Transcribes speech from an audio file using OpenAI's Whisper.
    Returns:
        str: Transcribed text from the audio
    """
    print(f"Loading Whisper model: {model_size}...")
    model = whisper.load_model(model_size)
    print(f"Transcribing: {audio_path}")
    
    result = model.transcribe(audio_path)
    
    print("Transcription complete.")
    return result["text"]
