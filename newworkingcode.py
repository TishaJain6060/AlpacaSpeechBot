# #!/usr/bin/env python3
# """
# GuideBot interaction pipeline

# Key TO KNOW for my dumbass:
# - Using `awaiting_command` flag
# - Core stuff:
#     * STT muting while TTS is speaking (prevents the stupid pc from hearing itself).
#     * Whisper stt.
#     * LLM JSON parsing.
# - Other:
#     * Wake word → prompt → next utterance → LLM → prints room + coords.
# """

# import queue
# import threading
# import time
# import re
# import json
# import sys
# import os
# import numpy as np
# import sounddevice as sd
# from faster_whisper import WhisperModel
# from piper import PiperVoice
# import google.generativeai as genai
# import wave

# # -----------------------------
# # CONFIGURATION
# # -----------------------------
# GENAI_API_KEY = "AIzaSyDJYYRtZPU00UDs-R7JkwOU5d8uIusZTx4"
# genai.configure(api_key=GENAI_API_KEY)

# WHISPER_MODEL = "small"
# PIPER_VOICE_FILE = "en_US-lessac-medium.onnx"

# SAMPLE_RATE = 16000
# CHUNK_DURATION = 3   # seconds per audio chunk
# CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
# RMS_THRESHOLD = 0.003

# COMMAND_START_TIMEOUT = 8      # seconds to start speaking after wake
# FOLLOWUP_IDLE_TIMEOUT = 10  

# ACTIVE_SESSION_TIMEOUT = 8

# #  wake phrases
# WAKE_PHRASES = [
#     "hey alpaca",
#     "alpaca",
#     "hello alpaca",
#     "can you help me",
#     "help",
#     "hey paca"
# ]

# COMMAND_TIMEOUT = 15 # do i need this? bruh?

# ROOM_COORDS = {
#     3140: [0.0, 17.706],
#     3141: [0.0, 17.706],
#     3150: [0.0, 8.743],
#     3160: [3.2, 7.1],
#     3161: [4.567, -4.880],
#     3170: [0.0, -9.543],
#     3171: [0.0, -9.543],
# }


# def get_coords(room_number):
#     return ROOM_COORDS.get(room_number, None)


# EXTRACTION_PROMPT = """
# You are a navigation interpreter. ONLY output JSON.

# Task:
# - Identify the target room number.
# - Convert number-words to digits.
# - Ignore irrelevant text.

# Output EXACTLY this format:
# {{
#   "target_room": <int or null>
# }}

# User said: "{utterance}"
# """


# # -----------------------------
# # UTILITY
# # -----------------------------
# def state(msg):
#     """For big state transitions/messages."""
#     print(f"\n>>> {msg}\n")
#     sys.stdout.flush()


# def debug(msg):
#     """For debug logging."""
#     print(f"[DEBUG] {msg}")
#     sys.stdout.flush()



# # -----------------------------
# # TTS WITH PIPER (Windows-safe, with STT mute)
# # -----------------------------
# class TTS:
#     def __init__(self, voice_file=PIPER_VOICE_FILE, bot_ref=None):
#         """
#         bot_ref: GuideBot instance (so we can toggle its stt_muted flag).
#         """
#         state("Loading Piper voice...")
#         self.voice = PiperVoice.load(voice_file)
#         self.bot_ref = bot_ref


#     def speak(self, text):
#         """
#         - Synthesizes text to a temp WAV file.
#         - Loads it fully into memory.
#         - Deletes the file (or tries).
#         - Plays from memory using sounddevice.
#         - Mutes STT during speak to avoid the robot hearing itself.
#         """
#         if not text.strip():
#             return

#         temp_file = "tts_output.wav"

#         # Mute STT so the robot doesn't hear its own voice
#         if self.bot_ref is not None:
#             self.bot_ref.stt_muted = True

#         # Generate WAV file
#         with wave.open(temp_file, "wb") as wf:
#             self.voice.synthesize_wav(text, wf)

#         # Read entire file into memory (this is just to avoid my Windows lock during playback)
#         with wave.open(temp_file, "rb") as wf:
#             data = wf.readframes(wf.getnframes())
#             framerate = wf.getframerate()

#         # Try to delete the temp file safely
#         try:
#             os.remove(temp_file)
#         except PermissionError:
#             # If Windows still holds a lock, just skip deletion;
#             # the file will be overwritten next time.
#             debug("Could not delete tts_output.wav (in use); skipping delete.")

#         # Convert to numpy and play
#         audio = np.frombuffer(data, dtype=np.int16)
#         sd.play(audio, framerate)
#         sd.wait()

#         # Small delay to avoid catching residual audio
#         time.sleep(0.15)

#         # Unmute STT
#         if self.bot_ref is not None:
#             self.bot_ref.stt_muted = False


# # -----------------------------
# # STT WITH WHISPER 
# # -----------------------------
# audio_queue = queue.Queue()


# class STT:
#     def __init__(self, model_name=WHISPER_MODEL, bot_ref=None):
#         state(f"Loading Whisper model ({model_name})...")
#         self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
#         self.bot_ref = bot_ref  # GuideBot, for checking stt_muted

#     def audio_callback(self, indata, frames, time_info, status):
#         """
#         Called by sounddevice in the audio thread whenever there's new mic data.
#         """
#         # If TTS is speaking, ignore input (prevents echo from robot itself)
#         if self.bot_ref is not None and getattr(self.bot_ref, "stt_muted", False):
#             return

#         if status:
#             print(f"Audio Status: {status}")

#         audio_data = np.squeeze(indata.copy())
#         rms = np.sqrt(np.mean(audio_data ** 2))

#         # Simple VAD: only queue audio if above RMS threshold
#         if rms > RMS_THRESHOLD:
#             audio_queue.put(audio_data)

#     def transcribe_audio_chunk(self, audio_chunk):
#         """
#         Transcribe one chunk of audio.
#         Set beam_size=1 and vad_filter=False to make it faster
#         """
#         segments, _ = self.model.transcribe(
#             audio_chunk,
#             beam_size=1,      # faster than beam_size=5
#             vad_filter=False,  # already using an RMS-based VAD
#             language="en"
#         )
#         text = " ".join(segment.text for segment in segments).strip()
#         return text.lower()

#     def run_transcription_loop(self, pipeline_handler, exit_flag):
#         state("Starting transcription loop...")
#         while not exit_flag.is_set():
#             try:
#                 audio_chunk = audio_queue.get(timeout=0.1)
#             except queue.Empty:
#                 # command start timeout + followup timeout checks live here
#                 pipeline_handler.tick()
#                 continue

#             # mark we got audio activity (used for timeouts)
#             pipeline_handler.on_audio_activity()

#             # tell user we're working (only in active/followup modes)
#             if pipeline_handler.awaiting_command or pipeline_handler.followup_mode:
#                 print("[STT] Heard audio — processing what you said...")
#                 sys.stdout.flush()

#             try:
#                 text = self.transcribe_audio_chunk(audio_chunk)
#             except Exception as e:
#                 print("[STT ERROR]", e)
#                 continue

#             if not text.strip():
#                 # if we were expecting something, prompt retry
#                 if pipeline_handler.awaiting_command or pipeline_handler.followup_mode:
#                     pipeline_handler.on_no_speech_detected()
#                 continue

#             print(f"[STT] Heard: {text}")
#             pipeline_handler.process_stt_result(text)


# # -----------------------------
# # LLM Client
# # -----------------------------
# class LLMClient:
#     def __init__(self):
#         self.model = genai.GenerativeModel("gemini-2.5-flash")

#     def extract_room(self, utterance):
#         """
#         Ask the LLM to extract a room number from the utterance.
#         Returns int room_number or None.
#         """
#         prompt = EXTRACTION_PROMPT.format(utterance=utterance)
#         try:
#             response = self.model.generate_content(prompt)
#             raw = response.text or ""
#             # Clean out code fences if any
#             raw = raw.replace("```json", "").replace("```", "").strip()
#             match = re.search(r"\{[\s\S]*\}", raw)
#             if not match:
#                 debug(f"LLM output had no JSON object: {raw}")
#                 return None
#             data = json.loads(match.group(0))
#             return data.get("target_room")
#         except Exception as e:
#             print("[LLM ERROR]", e)
#             return None

# def normalize_room_number(raw_number, valid_rooms):
#     """
#     Snap a noisy STT room number to the closest valid room.
#     """
#     try:
#         raw = int(round(float(raw_number)))
#     except ValueError:
#         return None

#     # Choose closest valid room
#     return min(valid_rooms, key=lambda r: abs(r - raw))


# # -----------------------------
# # Pipeline for GuideBot
# # -----------------------------
# class GuideBot:
#     def __init__(self):
#         # Flag used by STT to ignore mic input while TTS is speaking
#         self.stt_muted = False

#         # STT & TTS get a reference back to this GuideBot
#         self.stt = STT(bot_ref=self)
#         self.tts = TTS(bot_ref=self)
#         self.llm = LLMClient()

#         self.exit_flag = threading.Event()
#         self.awaiting_command = False
#         self.command_buffer = ""

#         self.last_wake_time = None
#         self.last_activity_time = time.time()
#         self.followup_mode = False
#         self.awaiting_yesno = False

#     def process_stt_result(self, text):
#         text = (text or "").strip().lower()
#         if not text:
#             return

#         self.last_activity_time = time.time()

#         # If we asked "Is that all?" then awaiting_command=True but we really want yes/no handling
#         if self.awaiting_command and ("is that all" in getattr(self, "last_bot_prompt", "")):
#             pass  # optional if you store prompts; not required

#         # If we're awaiting and user says yes/no, handle it
#         if self.awaiting_yesno:
#             if any(w in text for w in ["yes", "yeah", "yep", "that's all"]):
#                 self.tts.speak("Okay. Just say hey alpaca if you need anything else.")
#                 self.awaiting_yesno = False
#                 self.awaiting_command = False
#                 self.command_buffer = ""
#                 self.last_wake_time = None
#                 return

#             if any(w in text for w in ["no", "nope", "not yet"]):
#                 self.tts.speak("Okay. What else can I help you with?")
#                 self.awaiting_yesno = False
#                 self.awaiting_command = True
#                 self.command_buffer = ""
#                 self.last_wake_time = time.time()
#                 return

#             # If it's neither yes nor no, just treat it like a normal command:
#             self.awaiting_yesno = False
#             self.awaiting_command = True
#             self.command_buffer = ""

#         if not self.awaiting_command:
#             # PASSIVE: look for wake phrase
#             if any(w in text for w in WAKE_PHRASES):
#                 debug(f"Wake phrase detected in: {text!r}")
#                 self.tts.speak("How can I help you?")
#                 state("Listening for your navigation command...")
#                 self.awaiting_command = True
#                 self.followup_mode = False
#                 self.command_buffer = ""
#                 self.last_wake_time = time.time()
#                 self.last_activity_time = time.time()
#         else:
#             # ACTIVE: treat this chunk as the command
#             self.command_buffer += " " + text
#             utterance = self.command_buffer.strip()
#             debug(f"Captured command utterance: {utterance!r}")

#             self.handle_command(utterance)

#             # After fulfilling request: don't immediately go passive.
#             # Enter followup mode for up to 10s of silence; tick() will ask "Is that all?"
#             self.followup_mode = True
#             self.awaiting_command = True
#             self.command_buffer = ""
#             self.last_wake_time = None
#             self.last_activity_time = time.time()
        
#     def reset_to_passive(self, speak=True):
#         if speak:
#             self.tts.speak("Okay. Just say help me alpaca if you need anything else.")
#         self.awaiting_command = False
#         self.followup_mode = False
#         self.awaiting_yesno = False
#         self.command_buffer = ""
#         self.last_wake_time = None
#         self.last_activity_time = time.time()
#         state("Back to passive listening...")



#     def handle_command(self, utterance):
#         """
#         Given the command utterance, call LLM to get a room,
#         fetch coordinates, and speak + print navigation info.
#         """
#         if not utterance:
#             self.tts.speak("Sorry, I didn't catch that.")
#             return

#         # Get room number from LLM
#         target_room = self.llm.extract_room(utterance)

#         if not target_room:
#             self.tts.speak("Sorry, I didn't catch that. Can you try again?")
#             return

#         #  snap to closest valid room
#         normalized_room = normalize_room_number(
#             target_room,
#             ROOM_COORDS.keys()
#         )

#         if normalized_room != target_room:
#             debug(f"Normalized room {target_room} → {normalized_room}")

#         target_room = normalized_room



#         coords = get_coords(target_room)
#         if not coords:
#             self.tts.speak(f"Sorry, I don't know where room {target_room} is.")
#             print(f"Unknown room {target_room}")
#             return

#         # Confirm and print
#         self.tts.speak(f"How about I guide you to {target_room}.")
#         print("\n=== Navigation Command ===")
#         print(f"Destination Room: {target_room}")
#         print(f"Coordinates: {coords}")
#         print("==========================\n")
#         state("Listening for more requests (say another room, or wait)...")



#     def on_audio_activity(self):
#         self.last_activity_time = time.time()

#     def on_no_speech_detected(self):
#         # only annoy user if we were actually waiting for them
#         self.tts.speak("I didn't hear that. Can you try again?")

#     def tick(self):
#         """
#         Called frequently from STT loop when no audio arrives.
#         Handles timeouts without changing the core pipeline flow.
#         """
#         now = time.time()

#         if (self.awaiting_command or self.followup_mode or self.awaiting_yesno):
#             if (now - self.last_activity_time) > ACTIVE_SESSION_TIMEOUT:
#                 self.reset_to_passive(speak=True)
#                 return

#         # If we woke up but user never started speaking a command
#         if self.awaiting_command and self.last_wake_time is not None:
#             if (now - self.last_wake_time) > COMMAND_START_TIMEOUT and (now - self.last_activity_time) > COMMAND_START_TIMEOUT:
#                 self.tts.speak("I'm still listening. What room should I take you to?")
#                 # reset wake timer so we don't spam
#                 self.last_wake_time = now

#         # After fulfilling a request, we stay in followup_mode and wait
#         if self.followup_mode:
#             if (now - self.last_activity_time) > FOLLOWUP_IDLE_TIMEOUT:
#                 self.tts.speak("Is that all?")
#                 # After asking, exit followup_mode but keep awaiting_command to catch yes/no
#                 self.awaiting_yesno = True
#                 self.followup_mode = False
#                 self.awaiting_command = True
#                 self.command_buffer = ""
#                 self.last_wake_time = now


#     def run(self):
#         """
#         Main loop:
#         - Start audio stream
#         - Start STT thread
#         - Idle until Ctrl+C
#         """
#         stream = sd.InputStream(
#             samplerate=SAMPLE_RATE,
#             channels=1,
#             dtype="float32",
#             callback=self.stt.audio_callback,
#             blocksize=CHUNK_SIZE
#         )
#         stream.start()

#         stt_thread = threading.Thread(
#             target=self.stt.run_transcription_loop,
#             args=(self, self.exit_flag),
#             daemon=True
#         )
#         stt_thread.start()

#         state("Alpaca is running. Speak 'Help me, Alpaca', or 'Help'to wake me up!")

#         try:
#             while not self.exit_flag.is_set():
#                 time.sleep(0.1)
#         except KeyboardInterrupt:
#             print("Shutting down...")
#         finally:
#             self.exit_flag.set()
#             if stream.active:
#                 stream.stop()
#             print("Alpaca stopped.")


# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":
#     bot = GuideBot()
#     bot.run()
#!/usr/bin/env python3
import time
import queue
import csv
from datetime import datetime
import re
import os

import numpy as np
import sounddevice as sd
import soundfile as sf  # pip install soundfile
from faster_whisper import WhisperModel

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
BLOCK_DUR = 0.10                 # seconds per callback block (100ms)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DUR)

RMS_THRESHOLD = 0.003            # tune this per mic/environment
MIN_UTTERANCE_SEC = 0.35         # ignore tiny blips
END_SILENCE_SEC = 0.60           # how long of silence ends an utterance

WHISPER_MODEL = "small"          # or "base", "medium", etc.
LANGUAGE = "en"

CSV_PATH = "stt_metrics.csv"

# Accuracy prompt
PROMPT_FOR_REFERENCE = True

# Audio recording
SAVE_AUDIO = True
AUDIO_OUT_DIR = "recorded_audio"  # per-utterance WAVs will be saved here

# -----------------------------
# Audio queue: callback -> main thread
# -----------------------------
audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio status] {status}")
    audio = np.squeeze(indata.copy()).astype(np.float32)
    audio_q.put(audio)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

# -----------------------------
# Text normalization + metrics
# -----------------------------
_word_re = re.compile(r"[a-z0-9']+")

def normalize_for_wer(s: str) -> list[str]:
    s = (s or "").lower()
    return _word_re.findall(s)

def edit_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[m]

def wer(ref: str, hyp: str) -> float:
    r = normalize_for_wer(ref)
    h = normalize_for_wer(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / len(r)

def cer(ref: str, hyp: str) -> float:
    r = list((ref or "").lower())
    h = list((hyp or "").lower())
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(r, h) / len(r)

def ensure_csv_header():
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as _:
            return
    except FileNotFoundError:
        pass

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "audio_path",
            "audio_sec",
            "transcribe_sec",
            "end_to_text_sec",
            "start_to_text_sec",
            "rtf",
            "hypothesis",
            "reference",
            "wer",
            "cer",
        ])

def log_row(audio_path, audio_sec, transcribe_sec, end_to_text, start_to_text, rtf_val,
            hyp, ref, wer_val, cer_val):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            audio_path,
            f"{audio_sec:.3f}",
            f"{transcribe_sec:.3f}",
            f"{end_to_text:.3f}",
            f"{start_to_text:.3f}",
            f"{rtf_val:.3f}",
            hyp,
            ref,
            "" if wer_val is None else f"{wer_val:.4f}",
            "" if cer_val is None else f"{cer_val:.4f}",
        ])

# -----------------------------
# MAIN
# -----------------------------
def main():
    ensure_csv_header()
    if SAVE_AUDIO:
        os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

    print(f"Loading Whisper model: {WHISPER_MODEL} ...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("\nSTT-only test running.")
    print("Speak normally; pause to finalize an utterance. Ctrl+C to quit.\n")

    speaking = False
    buf = []
    last_voice_time = None
    speech_start_t = None  # perf_counter timestamp

    min_samples = int(MIN_UTTERANCE_SEC * SAMPLE_RATE)
    end_silence = END_SILENCE_SEC

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        try:
            while True:
                block = audio_q.get()
                level = rms(block)
                now = time.perf_counter()

                voice = level > RMS_THRESHOLD

                if voice:
                    if not speaking:
                        speaking = True
                        buf = []
                        speech_start_t = now
                        print("[VAD] speech start")
                    buf.append(block)
                    last_voice_time = now
                else:
                    if speaking:
                        buf.append(block)

                        if last_voice_time is not None and (now - last_voice_time) >= end_silence:
                            speaking = False
                            audio = np.concatenate(buf) if buf else np.array([], dtype=np.float32)

                            if audio.size < min_samples:
                                print("[VAD] ignored (too short)\n")
                                continue

                            # ---- Save utterance audio to WAV ----
                            audio_path = ""
                            if SAVE_AUDIO:
                                # safe filename timestamp
                                ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
                                audio_path = os.path.join(AUDIO_OUT_DIR, f"utterance_{ts}.wav")
                                # write 16-bit PCM WAV
                                sf.write(audio_path, audio, SAMPLE_RATE, subtype="PCM_16")

                            speech_end_t = time.perf_counter()
                            audio_sec = audio.size / SAMPLE_RATE

                            print("[VAD] speech end -> transcribing...")

                            t0 = time.perf_counter()
                            try:
                                segments, _ = model.transcribe(
                                    audio,
                                    language=LANGUAGE,
                                    beam_size=1,
                                    vad_filter=False,
                                )
                                hyp = " ".join(s.text for s in segments).strip()
                            except Exception as e:
                                print(f"[STT ERROR] {e}\n")
                                continue
                            t1 = time.perf_counter()

                            transcribe_sec = t1 - t0
                            end_to_text = t1 - speech_end_t
                            start_to_text = t1 - (speech_start_t if speech_start_t is not None else speech_end_t)
                            rtf_val = transcribe_sec / max(audio_sec, 1e-6)

                            if hyp:
                                print(f"[STT] {hyp}\n")
                            else:
                                print("[STT] (empty)\n")

                            print(
                                f"[LAT] audio={audio_sec:.2f}s | transcribe={transcribe_sec:.2f}s | "
                                f"end→text={end_to_text:.2f}s | start→text={start_to_text:.2f}s | RTF={rtf_val:.2f}\n"
                            )

                            ref = ""
                            wer_val = None
                            cer_val = None
                            if PROMPT_FOR_REFERENCE:
                                ref = input(
                                    "Type what you actually said (ground truth) or press Enter to skip:\n> "
                                ).strip()
                                if ref:
                                    wer_val = wer(ref, hyp)
                                    cer_val = cer(ref, hyp)
                                    print(f"[ACC] WER={wer_val:.3f} | CER={cer_val:.3f}\n")

                            log_row(
                                audio_path,
                                audio_sec,
                                transcribe_sec,
                                end_to_text,
                                start_to_text,
                                rtf_val,
                                hyp,
                                ref,
                                wer_val,
                                cer_val,
                            )

        except KeyboardInterrupt:
            print("\nStopping STT-only test.")
            print(f"Saved metrics to: {CSV_PATH}")
            if SAVE_AUDIO:
                print(f"Saved utterance WAVs to: {AUDIO_OUT_DIR}/")

if __name__ == "__main__":
    main()
