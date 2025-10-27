import queue
import sounddevice as sd
import numpy as np
import whisper
import threading
import time
import warnings
import os
import google.generativeai as genai

genai.configure(api_key="Key")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# --- CONFIG ---
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 3
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# --- Gemini Setup ---
# genai.configure(api_key=os.getenv("AIzaSyAwRx55yf-VQ1I4ycZT6dgxCe26dREuOzI"))
# gemini_model = genai.GenerativeModel("gemini-2.0-flash")  

# --- Whisper Setup ---
model = whisper.load_model("base")

audio_queue = queue.Queue()
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
last_time = time.time()

def generate_gemini_response(prompt: str):
    """Send the transcribed text to Gemini and print the response."""
    print("🤖 Thinking...")
    try:
        # response = gemini_model.generate_content(prompt)
        response = gemini_model.generate_content(prompt)
        print("-" * 50)
        print(f"🤖 GEMINI RESPONSE:\n{response.text}")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio_queue():
    global audio_buffer, last_time
    while True:
        chunk = audio_queue.get()
        chunk = np.squeeze(chunk)
        audio_buffer = np.roll(audio_buffer, -len(chunk))
        audio_buffer[-len(chunk):] = chunk

        # Throttle: only process every 2 seconds
        if time.time() - last_time < 2:
            continue
        last_time = time.time()

        result = model.transcribe(audio_buffer, fp16=False)
        text = result["text"].strip()
        if len(text) > 3:
            print(f"🗣️ User: {text}")
            audio_buffer[:] = 0
            threading.Thread(target=generate_gemini_response, args=(text,), daemon=True).start()

def main():
    print("🎙️ Listening... Speak into your mic.")
    threading.Thread(target=process_audio_queue, daemon=True).start()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
