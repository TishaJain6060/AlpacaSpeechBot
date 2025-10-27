import queue
import sounddevice as sd
import numpy as np
import whisper
import threading
import time
import warnings
# --- NEW: Gemini Imports ---
import os
from google import genai
from google.genai.errors import APIError
# ---------------------------

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# --- NEW: Gemini Client Setup ---
# It is recommended to set your API key as an environment variable (GEMINI_API_KEY)
try:
    client = genai.Client(api_key="Key")
    #client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure your GEMINI_API_KEY environment variable is set correctly.")
    client = None # Set to None if initialization fails
# --------------------------------

# Load Whisper model
model = whisper.load_model("tiny")

# Audio settings
SAMPLE_RATE = 16000        # Whisper works best at 16 kHz
CHUNK_DURATION = 0.5      # seconds per mic capture chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 3       # seconds of audio buffer for context
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

audio_queue = queue.Queue()
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# --- NEW: Function to Call Gemini API ---
def generate_gemini_response(prompt: str):
    """Sends the transcribed text to Gemini and prints the response."""
    if not client:
        print("🤖 ERROR: Gemini client not initialized. Cannot generate response.")
        return

    print("🤖 Thinking...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', # A fast and capable model for chat/text
            contents=prompt
        )
        print("-" * 50)
        print(f"🤖 **GEMINI RESPONSE**:\n{response.text}")
        print("-" * 50)
        
    except APIError as e:
        print(f"🤖 API Error: {e}")
    except Exception as e:
        print(f"🤖 An unexpected error occurred: {e}")

# ------------------------------------------

def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio_queue():
    """Continuously process audio from the queue in near real-time"""
    global audio_buffer
    try:
        while True:
            chunk = audio_queue.get()
            chunk = np.squeeze(chunk)

            # Shift buffer and append new chunk
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            # Only transcribe if buffer is "filled" enough
            if np.any(audio_buffer != 0):
                # Whisper expects float32 in [-1,1]
                result = model.transcribe(audio_buffer, fp16=False, language=None)
                text = result["text"].strip()
                
                # --- MODIFIED: Gemini Integration Point ---
                if text:
                    print(f"🗣️ User: {text}")
                    # Clear the audio buffer after a successful transcription
                    # to prevent repeated processing of the same speech.
                    audio_buffer[:] = 0 
                    
                    # 🚀 THIS IS WHERE THE GEMINI API IS CALLED 🚀
                    # We run the response generation in a new thread to avoid blocking the audio queue
                    threading.Thread(target=generate_gemini_response, args=(text,), daemon=True).start()
                # -------------------------------------------

    except KeyboardInterrupt:
        print("\n⏹️ Stopped listening. Goodbye!")

def main():
    if not client:
        return

    print("🎙️ Listening... Speak into your mic.")
    
    # Start the transcription and Gemini generation thread
    threading.Thread(target=process_audio_queue, daemon=True).start()
    
    # Open microphone stream
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=CHUNK_SIZE
    ):
        while True:
            time.sleep(0.1)  # Keep main thread alive

if __name__ == "__main__":
    main()