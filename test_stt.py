#!/usr/bin/env python3
import time
import queue
import os
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
BLOCK_DUR = 0.10                 # seconds per callback block
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DUR)

RMS_THRESHOLD = 0.003            # tune this
MIN_UTTERANCE_SEC = 0.35         # ignore tiny blips
END_SILENCE_SEC = 0.60           # how long of silence ends an utterance

WHISPER_MODEL = "small"          # or "base", "medium"
LANGUAGE = "en"


RECORD_SESSION_AUDIO = True
SESSION_OUT_DIR = "recorded_audio"
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

def save_wav_float32_mono(path: str, audio_f32: np.ndarray, sample_rate: int):
    """Save float32 mono audio in [-1, 1] to a 16-bit PCM WAV (stdlib only)."""
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

def main():
    if RECORD_SESSION_AUDIO:
        os.makedirs(SESSION_OUT_DIR, exist_ok=True)

    print(f"Loading Whisper model: {WHISPER_MODEL} ...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("\nSTT-only test running.")
    print("Speak normally; pause to finalize an utterance. Ctrl+C to quit.\n")

    speaking = False
    buf = []
    last_voice_time = None

    # holds ALL microphone audio for the entire run (continuous recording)
    all_audio = []

    min_samples = int(MIN_UTTERANCE_SEC * SAMPLE_RATE)
    end_silence = END_SILENCE_SEC

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )

    try:
        stream.start()

        while True:
            block = audio_q.get()  # blocking

            #store every block 
            if RECORD_SESSION_AUDIO:
                all_audio.append(block)

            level = rms(block)
            now = time.time()
            voice = level > RMS_THRESHOLD

            if voice:
                if not speaking:
                    speaking = True
                    buf = []
                    print("[VAD] speech start")
                buf.append(block)
                last_voice_time = now
            else:
                # still collect a tiny bit of trailing silence while "speaking"
                if speaking:
                    buf.append(block)

                    # if we've been silent long enough, finalize utterance
                    if last_voice_time is not None and (now - last_voice_time) >= end_silence:
                        speaking = False
                        audio = np.concatenate(buf) if buf else np.array([], dtype=np.float32)

                        if audio.size < min_samples:
                            print("[VAD] ignored (too short)\n")
                            continue

                        print("[VAD] speech end -> transcribing...")

                        try:
                            segments, _ = model.transcribe(
                                audio,
                                language=LANGUAGE,
                                beam_size=1,       
                                vad_filter=False, 
                            )
                            text = " ".join(s.text for s in segments).strip()
                        except Exception as e:
                            print(f"[STT ERROR] {e}\n")
                            continue

                        if text:
                            print(f"[STT] {text}\n")
                        else:
                            print("[STT] (empty)\n")

    except KeyboardInterrupt:
        print("\nStopping STT-only test (Ctrl+C).")
    finally:
        try:
            if stream.active:
                stream.stop()
        except Exception:
            pass

        if RECORD_SESSION_AUDIO and all_audio:
            full = np.concatenate(all_audio)
            ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")
            out_path = os.path.join(SESSION_OUT_DIR, f"full_session_{ts}.wav")
            save_wav_float32_mono(out_path, full, SAMPLE_RATE)
            print(f"[REC] saved full session audio: {out_path}")
        elif RECORD_SESSION_AUDIO:
            print("[REC] No audio captured; nothing to save.")

if __name__ == "__main__":
    main()

