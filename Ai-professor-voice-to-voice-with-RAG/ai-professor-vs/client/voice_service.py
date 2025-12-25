import os
from elevenlabs import generate, play, set_api_key
from dotenv import load_dotenv

load_dotenv()

# Set your ElevenLabs API Key in your .env file
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
set_api_key(ELEVEN_API_KEY)

def play_text_to_speech(text, voice_id="uE909sLRzIfE5XLq54rm"):
    """
    Generates and plays high-quality humanized audio using ElevenLabs.
    """
    try:
        # Use a professional voice like 'Brian' or 'Callum' for a professor feel
        audio = generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        
        # This handles the audio playback directly
        play(audio)
        
    except Exception as e:
        print(f"Error in ElevenLabs TTS: {e}")