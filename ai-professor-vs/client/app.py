import os
import wave
import pyaudio
import requests
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
from dotenv import load_dotenv

import voice_service as vs

# Load environment variables
load_dotenv()

# -------- CONFIG --------
RAG_URL = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8000/rag")
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium.en")
# ------------------------

def is_silence(data, threshold=50):
    return np.max(np.abs(data)) <= threshold

def record_audio(audio, stream):
    frames = []
    for _ in range(0, int(16000 / 1024 * 10)):
        frames.append(stream.read(1024))

    file_path = "temp.wav"
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    rate, data = wavfile.read(file_path)
    return None if is_silence(data) else file_path

def transcribe(model, file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join(seg.text for seg in segments)

def ask_rag(question):
    response = requests.post(RAG_URL, json={"question": question}, timeout=300)
    return response.json()["answer"]

def main():
    # ✅ CPU MODE (NO CUDA)
    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    print("AI Professor is listening...")

    while True:
        path = record_audio(audio, stream)
        if not path:
            continue

        text = transcribe(model, path)
        os.remove(path)

        if text.strip():
            print("You:", text)
            answer = ask_rag(text)
            print("AI Professor:", answer)
            vs.play_text_to_speech(answer)

if __name__ == "__main__":
    main()
