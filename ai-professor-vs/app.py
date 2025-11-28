import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

import voice_service as vs
# --- IMPORT FIX ---
# Import from the rag folder
from rag.AIVoiceAssistant import AIVoiceAssistant
# --- END IMPORT FIX ---

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10

ai_assistant = AIVoiceAssistant()


def is_silence(data, max_amplitude_threshold=50):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False

    

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription


def main():
    
    model_size = DEFAULT_MODEL_SIZE + ".en"
    
    # --- WHISPER DEVICE CHECK ---
    # If you do NOT have an NVIDIA GPU, change "cuda" to "cpu"
    # It will be slower, but it will work.
    # For slightly faster CPU, you can use compute_type="int8"
    model = WhisperModel(model_size, device="cuda",compute_type="int8_float16", num_workers=10)
    # model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=10)
    # Example for CPU:
    # model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=10)
    # --- END WHISPER DEVICE CHECK ---
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    customer_input_transcription = ""

    print("AI Professor is ready. Start speaking...")

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            
            # Record audio chunk
            print("_") # This prints every 10 seconds, meaning it's listening
            if not record_audio_chunk(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                if transcription and transcription.strip():
                    print(f"You: {transcription}")
                    
                    customer_input_transcription += "Customer: " + transcription + "\n"
                    
                    # Process customer input
                    output = ai_assistant.interact_with_llm(transcription)
                    
                    if output:
                        output = output.lstrip()
                        vs.play_text_to_speech(output)
                        print(f"AI Professor: {output}")
                else:
                    # Optional: Print this to know why it skipped
                    print("Debug: Transcription was empty, skipping LLM.")


    
    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()