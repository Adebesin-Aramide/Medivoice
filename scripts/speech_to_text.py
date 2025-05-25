import os
import whisper



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
