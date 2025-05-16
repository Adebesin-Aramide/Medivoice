import os
from pydub import AudioSegment
import speech_recognition as sr
from IPython.display import Audio

def convert_to_wav(input_path, output_path="converted.wav"):
    """
    Converts an audio file to WAV format using pydub.
    Supported: m4a, mp3, etc.
    """
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path

def speech_to_text(audio_file):
    """
    Transcribes speech from a WAV audio file using Google's Speech API.
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_text = r.record(source)  # record full audio
        try:
            print('Transcribing...')
            text = r.recognize_google(audio_text)
            print('Done.')
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google API; {e}"

