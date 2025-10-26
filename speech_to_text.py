import queue
import sounddevice as sd
import numpy as np
import whisper
import threading
import time
import openai
import warnings


# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load Whisper model
model = whisper.load_model("tiny")

# Audio settings
SAMPLE_RATE = 16000  # Whisper works best at 16 kHz
CHUNK_DURATION = 5   # seconds per chunk
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback function called by sounddevice for each audio chunk"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def process_audio_queue():
    """Continuously process audio chunks from the queue"""
    try:
        while True:
            audio_chunk = audio_queue.get()
            audio_data = np.squeeze(audio_chunk)

            # Whisper expects float32 in [-1, 1], which we already have
            result = model.transcribe(audio_data, fp16=False, language=None)
            text = result["text"].strip()
            if text:
                print(f"🗣️ {text}")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped listening. Goodbye!")

def main():
    print("🎙️ Listening... Speak into your mic.")
    
    # Start processing thread
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
