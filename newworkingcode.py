#!/usr/bin/env python3
"""
GuideBot interaction pipeline

Key TO KNOW for my dumbass:
- Using `awaiting_command` flag
- Core stuff:
    * STT muting while TTS is speaking (prevents the stupid pc from hearing itself).
    * Whisper stt.
    * LLM JSON parsing.
- Other:
    * Wake word → prompt → next utterance → LLM → prints room + coords.
"""

import queue
import threading
import time
import re
import json
import sys
import os
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
import google.generativeai as genai
import wave

# -----------------------------
# CONFIGURATION
# -----------------------------
GENAI_API_KEY = "Your Key"
genai.configure(api_key=GENAI_API_KEY)

WHISPER_MODEL = "small"
PIPER_VOICE_FILE = "en_US-lessac-medium.onnx"

SAMPLE_RATE = 16000
CHUNK_DURATION = 2   # seconds per audio chunk
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
RMS_THRESHOLD = 0.01

#  wake phrases
WAKE_PHRASES = [
    "hey alpaca",
    "alpaca",
    "hello alpaca",
    "can you help me",
    "help",
]

COMMAND_TIMEOUT = 15 # do i need this? bruh?

ROOM_COORDS = {
    3140: [0.0, 17.706],
    3141: [0.0, 17.706],
    3150: [0.0, 8.743],
    3160: [3.2, 7.1],
    3161: [4.567, -4.880],
    3170: [0.0, -9.543],
    3171: [0.0, -9.543],
}


def get_coords(room_number):
    return ROOM_COORDS.get(room_number, None)


EXTRACTION_PROMPT = """
You are a navigation interpreter. ONLY output JSON.

Task:
- Identify the target room number.
- Convert number-words to digits.
- Ignore irrelevant text.

Output EXACTLY this format:
{{
  "target_room": <int or null>
}}

User said: "{utterance}"
"""


# -----------------------------
# UTILITY
# -----------------------------
def state(msg):
    """For big state transitions/messages."""
    print(f"\n>>> {msg}\n")
    sys.stdout.flush()


def debug(msg):
    """For debug logging."""
    print(f"[DEBUG] {msg}")
    sys.stdout.flush()


# -----------------------------
# TTS WITH PIPER (Windows-safe, with STT mute)
# -----------------------------
class TTS:
    def __init__(self, voice_file=PIPER_VOICE_FILE, bot_ref=None):
        """
        bot_ref: GuideBot instance (so we can toggle its stt_muted flag).
        """
        state("Loading Piper voice...")
        self.voice = PiperVoice.load(voice_file)
        self.bot_ref = bot_ref

    def speak(self, text):
        """
        - Synthesizes text to a temp WAV file.
        - Loads it fully into memory.
        - Deletes the file (or tries).
        - Plays from memory using sounddevice.
        - Mutes STT during speak to avoid the robot hearing itself.
        """
        if not text.strip():
            return

        temp_file = "tts_output.wav"

        # Mute STT so the robot doesn't hear its own voice
        if self.bot_ref is not None:
            self.bot_ref.stt_muted = True

        # Generate WAV file
        with wave.open(temp_file, "wb") as wf:
            self.voice.synthesize_wav(text, wf)

        # Read entire file into memory (this is just to avoid my Windows lock during playback)
        with wave.open(temp_file, "rb") as wf:
            data = wf.readframes(wf.getnframes())
            framerate = wf.getframerate()

        # Try to delete the temp file safely
        try:
            os.remove(temp_file)
        except PermissionError:
            # If Windows still holds a lock, just skip deletion;
            # the file will be overwritten next time.
            debug("Could not delete tts_output.wav (in use); skipping delete.")

        # Convert to numpy and play
        audio = np.frombuffer(data, dtype=np.int16)
        sd.play(audio, framerate)
        sd.wait()

        # Small delay to avoid catching residual audio
        time.sleep(0.15)

        # Unmute STT
        if self.bot_ref is not None:
            self.bot_ref.stt_muted = False


# -----------------------------
# STT WITH WHISPER 
# -----------------------------
audio_queue = queue.Queue()


class STT:
    def __init__(self, model_name=WHISPER_MODEL, bot_ref=None):
        state(f"Loading Whisper model ({model_name})...")
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        self.bot_ref = bot_ref  # GuideBot, for checking stt_muted

    def audio_callback(self, indata, frames, time_info, status):
        """
        Called by sounddevice in the audio thread whenever there's new mic data.
        """
        # If TTS is speaking, ignore input (prevents echo from robot itself)
        if self.bot_ref is not None and getattr(self.bot_ref, "stt_muted", False):
            return

        if status:
            print(f"Audio Status: {status}")

        audio_data = np.squeeze(indata.copy())
        rms = np.sqrt(np.mean(audio_data ** 2))

        # Simple VAD: only queue audio if above RMS threshold
        if rms > RMS_THRESHOLD:
            audio_queue.put(audio_data)

    def transcribe_audio_chunk(self, audio_chunk):
        """
        Transcribe one chunk of audio.
        Set beam_size=1 and vad_filter=False to make it faster
        """
        segments, _ = self.model.transcribe(
            audio_chunk,
            beam_size=1,      # faster than beam_size=5
            vad_filter=False  # already using an RMS-based VAD
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text.lower()

    def run_transcription_loop(self, pipeline_handler, exit_flag):
        """
        Main loop for the STT thread:
        - Pull audio from queue,
        - Run transcription,
        - Pass text back into GuideBot.
        """
        state("Starting transcription loop...")
        while not exit_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                text = self.transcribe_audio_chunk(audio_chunk)
            except Exception as e:
                print("[STT ERROR]", e)
                continue

            if text:
                print(f"[STT] Heard: {text}")
                pipeline_handler.process_stt_result(text)


# -----------------------------
# LLM Client
# -----------------------------
class LLMClient:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def extract_room(self, utterance):
        """
        Ask the LLM to extract a room number from the utterance.
        Returns int room_number or None.
        """
        prompt = EXTRACTION_PROMPT.format(utterance=utterance)
        try:
            response = self.model.generate_content(prompt)
            raw = response.text or ""
            # Clean out code fences if any
            raw = raw.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{[\s\S]*\}", raw)
            if not match:
                debug(f"LLM output had no JSON object: {raw}")
                return None
            data = json.loads(match.group(0))
            return data.get("target_room")
        except Exception as e:
            print("[LLM ERROR]", e)
            return None


# -----------------------------
# Pipeline for GuideBot
# -----------------------------
class GuideBot:
    def __init__(self):
        # Flag used by STT to ignore mic input while TTS is speaking
        self.stt_muted = False

        # STT & TTS get a reference back to this GuideBot
        self.stt = STT(bot_ref=self)
        self.tts = TTS(bot_ref=self)
        self.llm = LLMClient()

        self.exit_flag = threading.Event()
        self.awaiting_command = False
        self.command_buffer = ""

    def process_stt_result(self, text):
        """
        Called by STT whenever we get a recognized chunk of speech.
        """
        if not self.awaiting_command:
            # We are in PASSIVE mode: look for wake phrase in the recognized text
            if any(w in text for w in WAKE_PHRASES):
                debug(f"Wake phrase detected in: {text!r}")
                self.tts.speak("How can I help you?")
                state("Listening for your navigation command...")
                self.awaiting_command = True
                self.command_buffer = ""
        else:
            # We are in ACTIVE mode: buffer this entire chunk as "the command"
            # (basically one chunk becomes the command)
            self.command_buffer += " " + text
            utterance = self.command_buffer.strip()
            debug(f"Captured command utterance: {utterance!r}")

            # Immediately handle the command (same as your original code)
            self.handle_command(utterance)

            # Reset back to passive listening
            self.awaiting_command = False
            self.command_buffer = ""

    def handle_command(self, utterance):
        """
        Given the command utterance, call LLM to get a room,
        fetch coordinates, and speak + print navigation info.
        """
        if not utterance:
            self.tts.speak("Sorry, I didn't catch that.")
            return

        # Get room number from LLM
        target_room = self.llm.extract_room(utterance)
        if not target_room:
            self.tts.speak("I could not find a room number in your request.")
            return

        coords = get_coords(target_room)
        if not coords:
            self.tts.speak(f"Sorry, I don't know where room {target_room} is.")
            print(f"Unknown room {target_room}")
            return

        # Confirm and print
        self.tts.speak(f"Okay, guiding to room {target_room}.")
        print("\n=== Navigation Command ===")
        print(f"Destination Room: {target_room}")
        print(f"Coordinates: {coords}")
        print("==========================\n")
        state("Returning to passive listening...")

    def run(self):
        """
        Main loop:
        - Start audio stream
        - Start STT thread
        - Idle until Ctrl+C
        """
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self.stt.audio_callback,
            blocksize=CHUNK_SIZE
        )
        stream.start()

        stt_thread = threading.Thread(
            target=self.stt.run_transcription_loop,
            args=(self, self.exit_flag),
            daemon=True
        )
        stt_thread.start()

        state("Alpaca is running. Speak 'Help me, Alpaca', or 'Help'to wake me up!")

        try:
            while not self.exit_flag.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.exit_flag.set()
            if stream.active:
                stream.stop()
            print("Alpaca stopped.")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    bot = GuideBot()
    bot.run()
