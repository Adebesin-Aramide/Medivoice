import os
import glob
from pydub import AudioSegment
import speech_recognition as sr
from IPython.display import Audio as IPyAudio
import easyocr

#OCR FUNCTION
def extract_text_from_latest_label(label_dir, lang='en', gpu=True):
    """
    Finds the latest 'label*.jpg' in the directory and extracts text using EasyOCR.
    """
    reader = easyocr.Reader([lang], gpu=gpu)
    pattern = os.path.join(label_dir, 'label*.jpg')
    candidates = glob.glob(pattern)

    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern}")
    
    latest = max(candidates, key=os.path.getctime)
    print("Running OCR on:", latest)

    results = reader.readtext(latest, detail=0, paragraph=True)
    print("\n=== EXTRACTED TEXT ===\n")
    for line in results:
        print(line)

    return results


#AUDIO CONVERSION FUNCTION 
def convert_to_wav(input_path, output_path="converted.wav"):
    """
    Converts an audio file (e.g., .m4a, .mp3) to WAV format using pydub.
    """
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path


#SPEECH-TO-TEXT FUNCTION
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


