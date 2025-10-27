import queue
import sounddevice as sd
import numpy as np
import whisper
import threading
import time
import warnings
import wave
from google import genai
from google.genai.types import GenerateContentConfig
from piper import PiperVoice

# =====================================
# CONFIG
# =====================================
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

SAMPLE_RATE = 16000
CHUNK_DURATION = 5
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Initialize modules
print("🔧 Loading Whisper model (tiny)...")
model = whisper.load_model("tiny")

print("🔧 Loading Piper voice...")
voice = PiperVoice.load("en_US-lessac-medium.onnx")

print("🔧 Initializing Gemini...")
gemini_client = genai.Client(api_key= "KEY")

audio_queue = queue.Queue()

# =====================================
# AUDIO STREAMING (STT)
# =====================================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def whisper_process():
    """Continuously process chunks from mic -> Whisper -> Gemini -> Piper"""
    while True:
        audio_chunk = audio_queue.get()
        audio_data = np.squeeze(audio_chunk)
        result = model.transcribe(audio_data, fp16=False, language=None)
        text = result["text"].strip()

        if text:
            print(f"\n🗣️ You said: {text}")
            threading.Thread(target=handle_llm_and_tts, args=(text,), daemon=True).start()

# =====================================
# GEMINI (LLM)
# =====================================
def generate_gemini_response(prompt: str) -> str:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=GenerateContentConfig(temperature=0.7)
        )
        text = response.text.strip()
        print(f"🤖 Gemini: {text}")
        return text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "Sorry, I ran into an issue generating a response."

# =====================================
# PIPER (TTS)
# =====================================
def speak_text(text: str):
    try:
        output_file = "gemini_reply.wav"
        with wave.open(output_file, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        with wave.open(output_file, "rb") as wf:
            data = wf.readframes(wf.getnframes())
            audio = np.frombuffer(data, dtype=np.int16)
            sd.play(audio, wf.getframerate())
            sd.wait()
        print("🔊 Finished speaking.")
    except Exception as e:
        print(f"❌ Piper error: {e}")

# =====================================
# PIPELINE HANDLER
# =====================================
def handle_llm_and_tts(user_text: str):
    """Run Gemini + TTS asynchronously for each detected phrase"""
    start = time.time()
    response = generate_gemini_response(user_text)
    mid = time.time()
    speak_text(response)
    end = time.time()
    print(f"⏱️ Gemini latency: {mid - start:.2f}s | TTS latency: {end - mid:.2f}s")

# =====================================
# MAIN
# =====================================
def main():
    print("🎙️ Listening... Speak into your mic (Ctrl+C to stop).")
    threading.Thread(target=whisper_process, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=CHUNK_SIZE,
    ):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped listening. Goodbye!")

if __name__ == "__main__":
    main()
