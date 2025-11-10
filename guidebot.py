import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import time
import warnings
import wave
import re
import base64
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types
from piper import PiperVoice
from io import BytesIO
from PIL import Image
import os
import io

# =====================================
# CONFIGURATION & GLOBAL STATE
# =====================================
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Audio Config
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
RMS_THRESHOLD = 0.015  # Volume threshold for noise gate (adjust as needed)

# Model Config
WHISPER_MODEL_NAME = "tiny"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
PIPER_VOICE_FILE = "en_US-lessac-medium.onnx"
GEMINI_API_KEY = "AIzaSyAwRx55yf-VQ1I4ycZT6dgxCe26dREuOzI"

# Map / Guide-bot config
MAP_FILE_PATH = "frb_second_floor.png"  # must exist

# Global Queue & State
audio_queue = queue.Queue()
map_display_queue = queue.Queue()   # optional: put base64 annotated images here
# Use a threading Event to signal the main loop to exit
exit_flag = threading.Event()
# Flag to indicate if the agent is currently confirming the exit
confirming_exit = False

# =====================================
# 1. AudioCapture Module
# =====================================
class AudioCapture:
    """Manages audio input stream and basic noise filtering."""
    def __init__(self, samplerate, blocksize, rms_threshold):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.rms_threshold = rms_threshold
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice to push audio to the queue."""
        if status:
            print(f"Audio Status: {status}")
        audio_data = np.squeeze(indata.copy())
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > self.rms_threshold:
            audio_queue.put(audio_data)

    def start_stream(self):
        """Starts the audio input stream."""
        print(f"🎙️ Starting audio stream at {self.samplerate}Hz...")
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=self.blocksize,
        )
        self.stream.start()

# =====================================
# 2. SpeechToText Module (Faster Whisper)
# =====================================
class SpeechToText:
    """Handles continuous transcription using faster-whisper."""
    def __init__(self, model_name):
        print(f"🔧 Loading faster-whisper model ({model_name})...")
        # device="cpu" is default, or can use "cuda" if available
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes a single chunk of audio."""
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=5,
            language=None,  # Auto-detect language
            vad_filter=True  # Use VAD
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text

    def run_transcription_loop(self, pipeline_handler):
        """Continuously pulls audio and transcribes it."""
        global confirming_exit

        print("🎧 Transcription process started.")
        while not exit_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # transcribe (blocking for the chunk)
            text = self.transcribe(audio_chunk)

            if text:
                print(f"\n🗣️ You said: {text}")

                if confirming_exit:
                    pipeline_handler.handle_exit_confirmation(text)
                else:
                    if pipeline_handler.is_farewell(text):
                        # set global confirming_exit so STT loop tracks state (baseline behavior)
                        confirming_exit = True
                        pipeline_handler.handle_farewell()
                    else:
                        # Spawn a thread to handle the LLM+TTS so STT loop keeps running
                        threading.Thread(
                            target=pipeline_handler.handle_conversation,
                            args=(text,),
                            daemon=True
                        ).start()

# =====================================
# 3A. LLMClient Module (Gemini chat)
# =====================================
class LLMClient:
    """Handles communication with the Gemini API for general chat responses."""
    def __init__(self, api_key, model_name):
        print("🔧 Initializing Gemini client...")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """Generates a response from Gemini (chat/general)."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(temperature=0.7)
            )
            text = response.text.strip()
            print(f"🤖 Gemini (chat): {text}")
            return text
        except Exception as e:
            print(f"❌ Gemini error (chat): {e}")
            return "Sorry, I ran into an issue generating a response."

# =====================================
# 3B. Multimodal LLM Client (Guide Bot)
# =====================================
class MultimodalLLMClient:
    """Gemini multimodal client: takes user text + map image and returns directions (text).
       Optionally can return annotated map as base64 if your prompts request it."""
    def __init__(self, api_key, model_name, map_file_path: str):
        print("🔧 Initializing Gemini Multimodal client (Navigation Mode)...")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.map_file_path = map_file_path
        self.map_image_part = None
        self._load_map_image()

    def _load_map_image(self):
        try:
            with open(self.map_file_path, "rb") as f:
                image_bytes = f.read()
            self.map_image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/png')
            print(f"🗺️ Map image '{self.map_file_path}' loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading map: {e}")
            self.map_image_part = None

    def generate_directions(self, user_text: str) -> dict:
        """
        Sends user_text + map image to Gemini. Returns dict:
            { "text": "<directions text>", "annotated_map_b64": "<optional base64 png>" }
        If map isn't available, returns a helpful error text.
        """
        if self.map_image_part is None:
            return {"text": "I cannot access the building map right now. Please check the map file.", "annotated_map_b64": None}

        system_instruction = (
            "You are a navigation assistant for the Ford Robotics Building's second floor. "
            "You are given a floor map and a user's request describing where they are and where they want to go. "
            "Return concise, step-by-step walking directions to reach the destination. Instructions should be"
            "natural, like a human would answer. "
        )
        #   "If possible include a short annotated map image as a base64 PNG in your multimodal output (if supported). "
           # "Return clear text directions first."

        try:
            # Provide the text and the map as contents
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[user_text, self.map_image_part],
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )

            # response.text is the textual directions
            directions_text = (response.text or "").strip()

            # Some Gemini multimodal outputs may include image parts; attempt to extract an image if present
            annotated_b64 = None
            # If the response contains parts or attachments (implementation depends on SDK),
            # you'd extract them here. We'll try a safe check:
            try:
                # some SDKs expose response.output_parts or similar; adapt if needed
                if hasattr(response, "parts") and response.parts:
                    for p in response.parts:
                        if getattr(p, "mime_type", "") == "image/png" and getattr(p, "data", None):
                            annotated_b64 = base64.b64encode(p.data).decode("utf-8")
                            break
            except Exception:
                annotated_b64 = None

            if not directions_text:
                directions_text = "Sorry, I couldn't determine the route. Please try again."

            return {"text": directions_text, "annotated_map_b64": annotated_b64}

        except Exception as e:
            print(f"❌ Gemini navigation error: {e}")
            return {"text": "Sorry, I ran into a technical issue while generating directions.", "annotated_map_b64": None}

# =====================================
# 4. TextToSpeech Module (Piper)
# =====================================
class TextToSpeech:
    """Handles text-to-speech synthesis using Piper (file-based approach)."""
    def __init__(self, voice_file):
        print("🔧 Loading Piper voice...")
        self.voice = PiperVoice.load(voice_file)

    def _clean_text(self, text: str) -> str:
        # Keep your previous cleaning rules (markdown removal etc.)
        text = text.replace('*', '').replace('_', '')
        text = re.sub(r'[\\/|\[\]{}()]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def speak(self, text: str):
        """Synthesizes text and plays the audio (writes a small WAV file then plays)."""
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            print("🔊 TTS skipped: cleaned text empty.")
            return

        print(f"🔊 Speaking text: {cleaned_text}")
        try:
            output_file = "gemini_reply.wav"
            # Synthesize to WAV file using the cleaned text
            with wave.open(output_file, "wb") as wav_file:
                self.voice.synthesize_wav(cleaned_text, wav_file)
            
            # Read and Play the WAV file
            with wave.open(output_file, "rb") as wf:
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)
                sd.play(audio, wf.getframerate())
                sd.wait() # Wait for playback to finish
            
            print("🔊 Finished speaking.")
        except Exception as e:
            print(f"❌ Piper error: {e}")

# =====================================
# 5. Generizable Pipeline Handler
# =====================================
class PipelineHandler:
    """Coordinates the LLM and TTS steps and manages exit and guide-bot routing."""
    def __init__(self, llm_client: LLMClient, mm_client: MultimodalLLMClient, tts_engine: TextToSpeech):
        self.llm_client = llm_client
        self.mm_client = mm_client
        self.tts_engine = tts_engine

        # Regex to detect common farewell phrases
        self.farewell_pattern = re.compile(r'\b(thank\s?you|thanks|cheers|bye|goodbye|that\'s\s?it)\b', re.IGNORECASE)
        self.affirmative_pattern = re.compile(r'\b(yes|yeah|yep|sure|continue)\b', re.IGNORECASE)
        self.negative_pattern = re.compile(r'\b(no|nope|nah|exit|stop)\b', re.IGNORECASE)

        # navigation intent patterns (room numbers, "go to", "directions", "where is")
        self.navigation_pattern = re.compile(r'\b(go to|directions to|directions|navigate to|where is|show me directions|how do i get to)\b', re.IGNORECASE)
        self.room_number_pattern = re.compile(r'\b(\d{3,4})\b')

    def is_farewell(self, text: str) -> bool:
        return bool(self.farewell_pattern.search(text))

    def looks_like_navigation(self, text: str) -> bool:
        # If user explicitly triggers navigation phrase OR contains two room numbers (src/dest) OR "go to room X"
        nav_phrase = bool(self.navigation_pattern.search(text))
        rooms = self.room_number_pattern.findall(text)
        go_to_room = bool(re.search(r'\bgo to (room )?\d{3,4}\b', text, re.IGNORECASE))
        return nav_phrase or len(rooms) >= 1 or go_to_room

    def handle_conversation(self, user_text: str):
        """Main conversation handler. Runs in a separate thread from STT loop."""
        start = time.time()
        try:
            if self.looks_like_navigation(user_text):
                # Use the multimodal guide-bot (map + text)
                result = self.mm_client.generate_directions(user_text)
                directions_text = result.get("text", "")
                annotated_b64 = result.get("annotated_map_b64")
                # If annotated map produced, push it to the display queue
                if annotated_b64:
                    map_display_queue.put(annotated_b64)
                # Speak directions
                self.tts_engine.speak(directions_text)
            else:
                # Fall back to chat LLM
                response = self.llm_client.generate_response(user_text)
                self.tts_engine.speak(response)
        except Exception as e:
            print(f"❌ Pipeline handler error: {e}")
            self.tts_engine.speak("Sorry, something went wrong while processing your request.")
        finally:
            print("\n🎧 Listening... Speak into your mic (say 'thank you' to stop).")
            pass

    def handle_farewell(self):
        """Starts exit confirmation flow (baseline behavior)."""
        confirmation_text = "Of course! Happy to help. Is there anything else I can assist you with?"
        self.tts_engine.speak(confirmation_text)

    def handle_exit_confirmation(self, user_text: str):
        global confirming_exit
        if self.negative_pattern.search(user_text):
            self.tts_engine.speak("You're welcome! Goodbye.")
            exit_flag.set()
        elif self.affirmative_pattern.search(user_text):
            confirming_exit = False
            self.tts_engine.speak("Great! What else can I help you with?")
        else:
            self.tts_engine.speak("I'm sorry, I didn't catch that. Do you need further assistance? Say 'yes' or 'no'.")

# =====================================
# Map display helper (optional)
# =====================================
def display_map_loop():
    """Saves base64 annotated maps to a file when posted to the map_display_queue."""
    print("🖥️ Map Display thread started.")
    while not exit_flag.is_set():
        try:
            b64_string = map_display_queue.get(timeout=0.1)
            try:
                imgdata = base64.b64decode(b64_string)
                img = Image.open(BytesIO(imgdata))
                out_name = "annotated_map.png"
                img.save(out_name)
                print(f"🖼️ Annotated map saved: {out_name}")
            except Exception as e:
                print(f"❌ Error saving annotated map: {e}")
        except queue.Empty:
            continue

# =====================================
# 🧪 MAIN EXECUTION
# =====================================
def main():
    # --- Initialization ---
    tts_engine = TextToSpeech(PIPER_VOICE_FILE)
    llm_client = LLMClient(GEMINI_API_KEY, GEMINI_MODEL_NAME)
    mm_client = MultimodalLLMClient(GEMINI_API_KEY, GEMINI_MODEL_NAME, MAP_FILE_PATH)
    stt_engine = SpeechToText(WHISPER_MODEL_NAME)

    pipeline_handler = PipelineHandler(llm_client, mm_client, tts_engine)

    audio_capture = AudioCapture(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        rms_threshold=RMS_THRESHOLD
    )

    # --- Start Processes ---
    audio_capture.start_stream()

    whisper_thread = threading.Thread(
        target=stt_engine.run_transcription_loop,
        args=(pipeline_handler,),
        daemon=True
    )
    whisper_thread.start()

    # optional map display thread
    threading.Thread(target=display_map_loop, daemon=True).start()

    print("\n🎧 Listening... Speak into your mic (Ctrl+C OR say 'thank you' to stop).")

    # --- Keep Main Thread Alive (Monitors exit_flag) ---
    try:
        exit_flag.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by KeyboardInterrupt.")
    finally:
        # Stop the stream
        try:
            if audio_capture.stream and audio_capture.stream.active:
                audio_capture.stream.stop()
        except Exception:
            pass
        print("\n⏹️ Voice assistant shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()
