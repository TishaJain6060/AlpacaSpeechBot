# # #!/usr/bin/env python3
# import queue
# import sounddevice as sd
# import numpy as np
# from faster_whisper import WhisperModel
# import threading
# import time
# import warnings
# import wave
# import re
# import os
# from google import genai
# from google.genai.types import GenerateContentConfig, Part
# from piper import PiperVoice

# # =====================================
# # CONFIGURATION & GLOBAL STATE
# # =====================================
# warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# # Audio Config
# SAMPLE_RATE = 16000
# CHUNK_DURATION = 10  # seconds
# CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
# RMS_THRESHOLD = 0.015

# # Model Config
# WHISPER_MODEL_NAME = "tiny"
# GEMINI_MODEL_NAME = "gemini-2.5-flash"
# PIPER_VOICE_FILE = "en_US-lessac-medium.onnx"
# GEMINI_API_KEY = "YOUR KEY"
# MAP_FILE_PATH = "octomapdraw.png"  # floorplan image path

# # Global queues & flags
# audio_queue = queue.Queue()
# llm_request_queue = queue.Queue()
# tts_queue = queue.Queue()
# exit_flag = threading.Event()
# confirming_exit = False

# # Regex
# FAREWELL_RE = re.compile(r'\b(thank\s?you|thanks|bye|goodbye|that\'s\s?it)\b', re.IGNORECASE)
# AFFIRMATIVE_RE = re.compile(r'\b(yes|yeah|yep|sure|continue)\b', re.IGNORECASE)
# NEGATIVE_RE = re.compile(r'\b(no|nope|nah|stop|exit)\b', re.IGNORECASE)
# NAVIGATION_RE = re.compile(r'\b(go to|directions to|navigate to|where is|room|how do i get to)\b', re.IGNORECASE)
# ROOM_RE = re.compile(r'\b(\d{3,4})\b')

# # =====================================
# # AUDIO CAPTURE
# # =====================================
# class AudioCapture:
#     def __init__(self, samplerate, blocksize, rms_threshold):
#         self.samplerate = samplerate
#         self.blocksize = blocksize
#         self.rms_threshold = rms_threshold

#     def audio_callback(self, indata, frames, time_info, status):
#         if status:
#             pass
#         audio_data = np.squeeze(indata.copy())
#         rms = np.sqrt(np.mean(audio_data**2))
#         if rms > self.rms_threshold:
#             try:
#                 audio_queue.put_nowait(audio_data)
#             except queue.Full:
#                 pass

#     def start_stream(self):
#         print(f" Starting audio stream at {self.samplerate}Hz...")
#         self.stream = sd.InputStream(
#             samplerate=self.samplerate,
#             channels=1,
#             dtype='float32',
#             callback=self.audio_callback,
#             blocksize=self.blocksize
#         )
#         self.stream.start()

# # =====================================
# # SPEECH TO TEXT (Whisper)
# # =====================================
# class SpeechToText:
#     def __init__(self, model_name):
#         print(f"Loading faster-whisper model ({model_name})...")
#         self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

#     def transcribe(self, audio_data: np.ndarray) -> str:
#         segments, info = self.model.transcribe(
#             audio_data, beam_size=5, language=None, vad_filter=True
#         )
#         text = " ".join(seg.text for seg in segments).strip()
#         return text

#     def run_transcription_loop(self, pipeline_handler):
#         global confirming_exit
#         print("Transcription thread running...")
#         while not exit_flag.is_set():
#             try:
#                 audio_chunk = audio_queue.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             text = self.transcribe(audio_chunk)
#             if text:
#                 print(f"\n You said: {text}")
#                 if confirming_exit:
#                     pipeline_handler.handle_exit_confirmation(text)
#                 else:
#                     if pipeline_handler.is_farewell(text):
#                         confirming_exit = True
#                         pipeline_handler.handle_farewell()
#                     else:
#                         threading.Thread(
#                             target=pipeline_handler.handle_conversation,
#                             args=(text,),
#                             daemon=True
#                         ).start()

# # =====================================
# # GEMINI MULTIMODAL CLIENT
# # =====================================
# class MultimodalNavigator:
#     def __init__(self, api_key, model_name, map_path):
#         print("Initializing Gemini client...")
#         self.client = genai.Client(api_key=api_key)
#         self.model_name = model_name
#         self.map_part = None
#         self._load_map(map_path)

#     def _load_map(self, path):
#         if not path or not os.path.exists(path):
#             print(f"Map file not found: {path}")
#             self.map_part = None
#             return
#         with open(path, "rb") as f:
#             img_bytes = f.read()
#         self.map_part = Part.from_bytes(data=img_bytes, mime_type="image/png")
#         print(f"Map loaded for context: {path}")

#     def generate_response(self, user_text: str) -> dict:
#         system_instruction = (
#             "You are an indoor navigation assistant. Given a floorplan image (if available) and a user's request "
#             "describing origin and destination, produce two sections separated by '---NAVIGATION-DATA---'.\n"
#             "Section 1 (Spoken): natural walking directions.\n"
#             "Section 2 (Navigation JSON): JSON array of actions: 'go_straight', 'turn_left', 'turn_right', 'reach_destination'."
#         )

#         contents = [user_text]
#         if self.map_part:
#             contents.append(self.map_part)

#         try:
#             response = self.client.models.generate_content(
#                 model=self.model_name,
#                 contents=contents,
#                 config=GenerateContentConfig(system_instruction=system_instruction, temperature=0.2)
#             )
#             text_out = getattr(response, "text", "") or str(response)
#             text_out = text_out.strip()

#             sep = "---NAVIGATION-DATA---"
#             if sep in text_out:
#                 spoken_part, nav_raw = text_out.split(sep, 1)
#                 spoken = spoken_part.strip()
#                 try:
#                     first_brace = nav_raw.index("[")
#                     last_brace = nav_raw.rindex("]") + 1
#                     nav_json = json.loads(nav_raw[first_brace:last_brace])
#                 except Exception:
#                     nav_json = nav_raw
#                 return {"spoken_text": spoken, "navigation_data": nav_json}
#             return {"spoken_text": text_out, "navigation_data": None}
#         except Exception as e:
#             print(f"Gemini error: {e}")
#             return {"spoken_text": "Sorry, I couldn't get directions.", "navigation_data": None}

# # =====================================
# # LLM CLIENT WRAPPER
# # =====================================
# class LLMClient:
#     def __init__(self, api_key, model_name, map_path):
#         self.navigator = MultimodalNavigator(api_key, model_name, map_path)

#     def generate_response(self, text):
#         return self.navigator.generate_response(text)

# # =====================================
# # TEXT TO SPEECH (PIPER)
# # =====================================
# class TextToSpeech:
#     def __init__(self, voice_file):
#         if os.path.exists(voice_file):
#             print("Loading Piper voice...")
#             self.voice = PiperVoice.load(voice_file)
#         else:
#             print("Piper voice file not found.")
#             self.voice = None

#     def _clean_text(self, text: str) -> str:
#         text = text.replace('*', '').replace('_', '')
#         text = re.sub(r'[\\/|\[\]{}()]', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def speak(self, text: str):
#         cleaned = self._clean_text(text)
#         if not cleaned:
#             return
#         print(f"{cleaned}")
#         if self.voice:
#             try:
#                 out_file = "tts_output.wav"
#                 with wave.open(out_file, "wb") as wf:
#                     self.voice.synthesize_wav(cleaned, wf)
#                 with wave.open(out_file, "rb") as wf:
#                     audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
#                     sd.play(audio, wf.getframerate())
#                     sd.wait()
#             except Exception as e:
#                 print(f"Piper TTS error: {e}")

# # =====================================
# # PIPELINE HANDLER
# # =====================================
# class PipelineHandler:
#     def __init__(self, llm_client: LLMClient, tts_engine: TextToSpeech):
#         self.llm_client = llm_client
#         self.tts_engine = tts_engine

#     def is_farewell(self, text: str) -> bool:
#         return bool(FAREWELL_RE.search(text))

#     def handle_conversation(self, user_text: str):
#         result = self.llm_client.generate_response(user_text)
#         spoken = result.get("spoken_text", "")
#         nav_data = result.get("navigation_data")
#         self.tts_engine.speak(spoken)
#         if nav_data:
#             print("Navigation JSON:", nav_data)

#     def handle_farewell(self):
#         self.tts_engine.speak("Of course! Happy to help. Is there anything else I can assist you with?")

#     def handle_exit_confirmation(self, user_text: str):
#         global confirming_exit
#         if NEGATIVE_RE.search(user_text):
#             self.tts_engine.speak("You're welcome! Goodbye.")
#             exit_flag.set()
#         elif AFFIRMATIVE_RE.search(user_text):
#             confirming_exit = False
#             self.tts_engine.speak("Great! What else can I help you with?")
#         else:
#             self.tts_engine.speak("I didn't catch that. Say 'yes' or 'no'.")

# # =====================================
# # MAIN EXECUTION
# # =====================================
# def main():
#     tts_engine = TextToSpeech(PIPER_VOICE_FILE)
#     llm_client = LLMClient(GEMINI_API_KEY, GEMINI_MODEL_NAME, MAP_FILE_PATH)
#     stt_engine = SpeechToText(WHISPER_MODEL_NAME)
#     pipeline_handler = PipelineHandler(llm_client, tts_engine)
#     audio_capture = AudioCapture(SAMPLE_RATE, CHUNK_SIZE, RMS_THRESHOLD)
    
#     # Start audio stream
#     audio_capture.start_stream()
    
#     # Start transcription thread
#     threading.Thread(
#         target=stt_engine.run_transcription_loop,
#         args=(pipeline_handler,),
#         daemon=True
#     ).start()

#     print("\nListening... Speak into your mic (Ctrl+C or say 'thank you' to stop).")
    
#     try:
#         exit_flag.wait()
#     except KeyboardInterrupt:
#         exit_flag.set()
#         print("\nStopped by KeyboardInterrupt.")
#     finally:
#         if audio_capture.stream.active:
#             audio_capture.stream.stop()
#         print("\nVoice assistant shutdown complete. Goodbye!")

# if __name__ == "__main__":
#     main()


import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import time
import warnings
import re
import os

# =====================================
# CONFIGURATION
# =====================================
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

SAMPLE_RATE = 16000
CHUNK_DURATION = 8
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
RMS_THRESHOLD = 0.015

WHISPER_MODEL_NAME = "tiny"

# Silence shutdown timeout
SILENCE_TIMEOUT = 15
last_voice_time = time.time()

# Wake word regex
WAKE_RE = re.compile(r"(hey alpaca|can you help me|hello alpaca)", re.IGNORECASE)

# Navigation regex
ROOM_RE = re.compile(r"\b(\d{3,4})\b")

# Room → coordinate map
ROOM_COORDS = {
    "3170": [0, -9.543],
    "3171": [0, -9.543],

    "3161": [4.567, -4.880],

    "3150": [0, 8.743],

    "3140": [0, 17.706],
    "3141": [0, 17.706],
}

# Global program state
audio_queue = queue.Queue()
exit_flag = threading.Event()
listening_active = False  # becomes True after wake word



# =====================================
# AUDIO CAPTURE
# =====================================
class AudioCapture:
    def __init__(self):
        pass

    def audio_callback(self, indata, frames, time_info, status):
        global last_voice_time
        audio_data = np.squeeze(indata.copy())
        rms = np.sqrt(np.mean(audio_data**2))

        if rms > RMS_THRESHOLD:
            last_voice_time = time.time()   # reset silence timer
            try:
                audio_queue.put_nowait(audio_data)
            except:
                pass

    def start(self):
        print("Starting audio capture...")
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE,
        )
        self.stream.start()



# =====================================
# SPEECH TO TEXT
# =====================================
class SpeechToText:
    def __init__(self):
        print("Loading Whisper tiny...")
        self.model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")

    def transcribe(self, audio_data):
        segments, _ = self.model.transcribe(audio_data, beam_size=5, vad_filter=True)
        return " ".join(seg.text for seg in segments).strip()

    def run(self, pipeline_handler):
        global listening_active

        while not exit_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text = self.transcribe(audio_chunk)
            if not text:
                continue

            print(f"> You said: {text}")

            # Not awakened yet → look for wake word
            if not listening_active:
                if WAKE_RE.search(text):
                    listening_active = True
                    print("Wake-word detected. Assistant is now listening.")
                continue

            # Already awake → handle navigation
            threading.Thread(
                target=pipeline_handler.handle_text,
                args=(text,),
                daemon=True
            ).start()



# =====================================
# PIPELINE HANDLER
# =====================================
class PipelineHandler:
    def __init__(self):
        pass

    def extract_rooms(self, text):
        rooms = ROOM_RE.findall(text)
        if len(rooms) >= 2:
            return rooms[0], rooms[1]
        return None, None

    def handle_text(self, text):
        origin, dest = self.extract_rooms(text)

        if not origin or not dest:
            print("Could not detect two room numbers.")
            return

        if origin not in ROOM_COORDS or dest not in ROOM_COORDS:
            print(" One of the rooms is not in the known map.")
            return

        print("\n===== NAVIGATION REQUEST =====")
        print(f"Origin room: {origin} → coords {ROOM_COORDS[origin]}")
        print(f"Destination room: {dest} → coords {ROOM_COORDS[dest]}")
        print("================================\n")



# =====================================
# MAIN
# =====================================
def main():
    global last_voice_time

    audio = AudioCapture()
    stt = SpeechToText()
    pipeline = PipelineHandler()

    audio.start()

    threading.Thread(
        target=stt.run,
        args=(pipeline,),
        daemon=True
    ).start()

    print("\nSay 'Hey Alpaca' to activate me. I will shut down after silence.\n")

    try:
        while not exit_flag.is_set():
            if time.time() - last_voice_time > SILENCE_TIMEOUT:
                print("\n🛑 Shutting down due to inactivity.")
                exit_flag.set()
            time.sleep(0.2)
    finally:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
