import os
import wave
import pyaudio
import requests
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from pynput import keyboard  # New library for key detection

import voice_service as vs

# Load environment variables
load_dotenv()

# -------- CONFIG --------
RAG_URL = os.getenv("RAG_SERVER_URL", "http://127.0.0.1:8000/rag")
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium.en")
# ------------------------

# Global variable to track spacebar state
is_recording = False

def on_press(key):
    global is_recording
    if key == keyboard.Key.space:
        is_recording = True

def on_release(key):
    global is_recording
    if key == keyboard.Key.space:
        is_recording = False
        return False  # Stop listener after release to process audio

def record_audio_ptt(audio, stream):
    """Records audio only while the spacebar is held down."""
    frames = []
    print("\n[ HOLD SPACE TO SPEAK ]")
    
    # Start keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        # Wait until space is pressed to start collecting frames
        while not is_recording:
            pass
        
        print(">>> Recording... Release space to stop.")
        
        # Collect audio frames while space is held
        while is_recording:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        print(">>> Stopped recording. Processing...")
        listener.stop()

    file_path = "temp.wav"
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    return file_path

def transcribe(model, file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join(seg.text for seg in segments)

def ask_rag(question):
    try:
        response = requests.post(RAG_URL, json={"question": question}, timeout=300)
        return response.json().get("answer", "I couldn't find an answer.")
    except Exception as e:
        return f"Error connecting to RAG server: {e}"

def main():
    # ✅ CPU MODE
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

    try:
        while True:
            path = record_audio_ptt(audio, stream)
            
            text = transcribe(model, path)
            if os.path.exists(path):
                os.remove(path)

            if text.strip():
                print(f"You: {text}")
                answer = ask_rag(text)
                print(f"AI Professor: {answer}")
                vs.play_text_to_speech(answer)
            else:
                print("No speech detected.")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()