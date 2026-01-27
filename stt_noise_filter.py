# #!/usr/bin/env python3
# import time
# import queue
# import os
# import wave
# from datetime import datetime

# import numpy as np
# import sounddevice as sd
# from faster_whisper import WhisperModel

# from scipy.signal import butter, filtfilt, iirnotch

# # -----------------------------
# # CONFIG
# # -----------------------------
# SAMPLE_RATE = 16000
# BLOCK_DUR = 0.10
# BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DUR)

# RMS_THRESHOLD = 0.003
# MIN_UTTERANCE_SEC = 0.35
# END_SILENCE_SEC = 0.60

# WHISPER_MODEL = "small"
# LANGUAGE = "en"

# # Recording
# RECORD_SESSION_AUDIO = True
# SESSION_OUT_DIR = "recorded_audio26"
# SAVE_RAW_AND_CLEAN = True  # saves raw + filtered full-session wavs

# # Filtering pipeline
# APPLY_BANDPASS = True
# BP_LOW_HZ = 80.0
# BP_HIGH_HZ = 7000.0
# BP_ORDER = 4

# APPLY_MULTI_NOTCH = True
# CALIBRATE_SEC = 3.0          # stay quiet at start to learn tones
# PEAK_SEARCH_LO = 20.0
# PEAK_SEARCH_HI = 140.0
# MAX_NOTCH_PEAKS = 6          # notch top N tones
# MIN_PEAK_SEP_HZ = 4.0        # keep notches separated
# NOTCH_Q = 35.0               # higher = narrower notch
# HARMONICS = 3                # notch base + 2x (keep small to protect speech)

# # -----------------------------
# # Audio queue: callback -> main thread
# # -----------------------------
# audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print(f"[Audio status] {status}")
#     audio = np.squeeze(indata.copy()).astype(np.float32)
#     audio_q.put(audio)

# def rms(x: np.ndarray) -> float:
#     return float(np.sqrt(np.mean(x * x) + 1e-12))

# def save_wav_float32_mono(path: str, audio_f32: np.ndarray, sample_rate: int):
#     """Save float32 mono audio in [-1,1] to 16-bit PCM WAV."""
#     audio_i16 = np.clip(audio_f32, -1.0, 1.0)
#     audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
#     with wave.open(path, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate)
#         wf.writeframes(audio_i16.tobytes())

# # -----------------------------
# # Filters
# # -----------------------------
# def bandpass_filter(x: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
#     """Butterworth band-pass (speech-safe)."""
#     x = x.astype(np.float32)
#     nyq = 0.5 * sr
#     low = max(low_hz / nyq, 1e-6)
#     high = min(high_hz / nyq, 0.999999)
#     if not (0 < low < high < 1):
#         return x
#     b, a = butter(order, [low, high], btype="band")
#     return filtfilt(b, a, x).astype(np.float32)

# def notch_once(x: np.ndarray, sr: int, freq: float, q: float) -> np.ndarray:
#     """Single iirnotch + zero-phase filtfilt."""
#     b, a = iirnotch(w0=freq, Q=q, fs=sr)
#     return filtfilt(b, a, x).astype(np.float32)

# def apply_multi_notch(x: np.ndarray, sr: int, freqs: list[float], q: float, harmonics: int) -> np.ndarray:
#     """Apply notches at freqs and optional harmonics."""
#     y = x.astype(np.float32)
#     nyq = 0.5 * sr
#     for f0 in freqs:
#         for k in range(1, harmonics + 1):
#             fk = f0 * k
#             if fk >= nyq - 5:
#                 break
#             y = notch_once(y, sr, fk, q)
#     return y

# def find_tonal_peaks(x: np.ndarray, sr: int, f_lo: float, f_hi: float,
#                      max_peaks: int, min_sep_hz: float) -> list[float]:
#     """
#     Find strongest tonal peaks in [f_lo, f_hi] via FFT peak-pick with separation.
#     Best when x is mostly 'silence' (robot noise only).
#     """
#     x = x.astype(np.float32)
#     x = x - np.mean(x)

#     # pad to at least 1s for resolution
#     if len(x) < sr:
#         x = np.pad(x, (0, sr - len(x)))

#     n = len(x)
#     win = np.hanning(n).astype(np.float32)
#     spec = np.fft.rfft(x * win)
#     freqs = np.fft.rfftfreq(n, 1.0 / sr)
#     power = (np.abs(spec) ** 2)

#     mask = (freqs >= f_lo) & (freqs <= f_hi)
#     freqs_b = freqs[mask]
#     power_b = power[mask]
#     if freqs_b.size == 0:
#         return []

#     order = np.argsort(power_b)[::-1]
#     peaks: list[float] = []
#     for idx in order:
#         f = float(freqs_b[idx])
#         if all(abs(f - pf) >= min_sep_hz for pf in peaks):
#             peaks.append(f)
#         if len(peaks) >= max_peaks:
#             break
#     return peaks

# def enhance_audio(x: np.ndarray, sr: int, notch_freqs: list[float]) -> np.ndarray:
#     """Your full denoise pipeline: band-pass then multi-notch."""
#     y = x.astype(np.float32)

#     if APPLY_BANDPASS:
#         y = bandpass_filter(y, sr, BP_LOW_HZ, BP_HIGH_HZ, order=BP_ORDER)

#     if APPLY_MULTI_NOTCH and notch_freqs:
#         y = apply_multi_notch(y, sr, notch_freqs, q=NOTCH_Q, harmonics=HARMONICS)

#     return y

# # -----------------------------
# # MAIN
# # -----------------------------
# def main():
#     if RECORD_SESSION_AUDIO:
#         os.makedirs(SESSION_OUT_DIR, exist_ok=True)

#     print(f"Loading Whisper model: {WHISPER_MODEL} ...")
#     model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

#     print("\nWhisper STT test + (band-pass + learned multi-notch). Ctrl+C to quit.\n")

#     speaking = False
#     buf: list[np.ndarray] = []
#     last_voice_time = None

#     all_audio_raw: list[np.ndarray] = []
#     all_audio_clean: list[np.ndarray] = []

#     min_samples = int(MIN_UTTERANCE_SEC * SAMPLE_RATE)

#     # --- calibration ---
#     notch_freqs: list[float] = []
#     calib_buf: list[np.ndarray] = []
#     calib_needed = int(CALIBRATE_SEC * SAMPLE_RATE)

#     stream = sd.InputStream(
#         samplerate=SAMPLE_RATE,
#         channels=1,
#         dtype="float32",
#         blocksize=BLOCK_SIZE,
#         callback=audio_callback,
#     )

#     try:
#         stream.start()

#         if APPLY_MULTI_NOTCH:
#             print(f"[CAL] Stay quiet for ~{CALIBRATE_SEC:.1f}s so I can learn the robot’s tones...\n")

#         while True:
#             block = audio_q.get()

#             # Record raw session
#             if RECORD_SESSION_AUDIO:
#                 all_audio_raw.append(block)

#             # Learn tonal peaks from initial "silence"
#             if APPLY_MULTI_NOTCH and not notch_freqs:
#                 calib_buf.append(block)
#                 if sum(b.size for b in calib_buf) >= calib_needed:
#                     calib_audio = np.concatenate(calib_buf)[:calib_needed]

#                     # Apply band-pass before peak finding (helps focus on relevant range)
#                     if APPLY_BANDPASS:
#                         calib_audio_for_peaks = bandpass_filter(calib_audio, SAMPLE_RATE, BP_LOW_HZ, BP_HIGH_HZ, order=BP_ORDER)
#                     else:
#                         calib_audio_for_peaks = calib_audio

#                     notch_freqs = find_tonal_peaks(
#                         calib_audio_for_peaks, SAMPLE_RATE,
#                         PEAK_SEARCH_LO, PEAK_SEARCH_HI,
#                         max_peaks=MAX_NOTCH_PEAKS,
#                         min_sep_hz=MIN_PEAK_SEP_HZ,
#                     )

#                     print(f"[CAL] Notching tones (Hz): {[round(f, 1) for f in notch_freqs]}")
#                     print(f"[CAL] Band-pass {BP_LOW_HZ:.0f}-{BP_HIGH_HZ:.0f} Hz | Q={NOTCH_Q} | harmonics={HARMONICS}\n")
#                     print("[INFO] You can speak now.\n")
#                 continue

#             # For “cleaned session recording”, filter block-by-block (lightweight)
#             if RECORD_SESSION_AUDIO and SAVE_RAW_AND_CLEAN:
#                 all_audio_clean.append(enhance_audio(block, SAMPLE_RATE, notch_freqs))

#             level = rms(block)
#             now = time.time()
#             voice = level > RMS_THRESHOLD

#             if voice:
#                 if not speaking:
#                     speaking = True
#                     buf = []
#                     print("[VAD] speech start")
#                 buf.append(block)
#                 last_voice_time = now
#             else:
#                 if speaking:
#                     buf.append(block)

#                     if last_voice_time is not None and (now - last_voice_time) >= END_SILENCE_SEC:
#                         speaking = False
#                         audio = np.concatenate(buf) if buf else np.array([], dtype=np.float32)

#                         if audio.size < min_samples:
#                             print("[VAD] ignored (too short)\n")
#                             continue

#                         # ---- Filter utterance BEFORE Whisper ----
#                         audio_clean = enhance_audio(audio, SAMPLE_RATE, notch_freqs)

#                         print("[VAD] speech end -> transcribing...")

#                         try:
#                             segments, _ = model.transcribe(
#                                 audio_clean,
#                                 language=LANGUAGE,
#                                 beam_size=1,
#                                 vad_filter=False,
#                             )
#                             text = " ".join(s.text for s in segments).strip()
#                         except Exception as e:
#                             print(f"[STT ERROR] {e}\n")
#                             continue

#                         print(f"[STT] {text}\n" if text else "[STT] (empty)\n")

#     except KeyboardInterrupt:
#         print("\nStopping (Ctrl+C).")

#     finally:
#         try:
#             if stream.active:
#                 stream.stop()
#         except Exception:
#             pass

#         if RECORD_SESSION_AUDIO and all_audio_raw:
#             ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")

#             raw = np.concatenate(all_audio_raw)
#             raw_path = os.path.join(SESSION_OUT_DIR, f"full_session_RAW_{ts}.wav")
#             save_wav_float32_mono(raw_path, raw, SAMPLE_RATE)
#             print(f"[REC] saved RAW session:   {raw_path}")

#             if SAVE_RAW_AND_CLEAN:
#                 if all_audio_clean:
#                     clean = np.concatenate(all_audio_clean)
#                 else:
#                     # fallback: clean after the fact
#                     clean = enhance_audio(raw, SAMPLE_RATE, notch_freqs)

#                 clean_path = os.path.join(SESSION_OUT_DIR, f"full_session_CLEAN_{ts}.wav")
#                 save_wav_float32_mono(clean_path, clean, SAMPLE_RATE)
#                 print(f"[REC] saved CLEAN session: {clean_path}")

#         elif RECORD_SESSION_AUDIO:
#             print("[REC] No audio captured; nothing to save.")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import time
import queue
import os
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from webrtc_noise_gain import AudioProcessor  # pip install webrtc-noise-gain

# -----------------------------
# CONFIG
# -----------------------------
SAMPLE_RATE = 16000
BLOCK_DUR = 0.10                 # 100ms callback blocks
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DUR)

# You can usually LOWER this once NS is enabled because noise floor drops.
RMS_THRESHOLD = 0.003
MIN_UTTERANCE_SEC = 0.35
END_SILENCE_SEC = 0.60

WHISPER_MODEL = "small"
LANGUAGE = "en"

# Recording
RECORD_SESSION_AUDIO = True
SESSION_OUT_DIR = "recorded_audio262"
SAVE_RAW_AND_CLEAN = True

# WebRTC processing
# noise_suppression_level: 0=off, 4=max
NOISE_SUPPRESSION_LEVEL = 3
# auto_gain_dbfs: 0=off, else [0..31] (I recommend OFF first; Whisper doesn't need AGC)
AUTO_GAIN_DBFS = 0

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
    """Save float32 mono audio in [-1, 1] to 16-bit PCM WAV."""
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

def float32_to_int16(audio_f32: np.ndarray) -> np.ndarray:
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    return (audio_i16 * 32767.0).astype(np.int16)

def int16_to_float32(audio_i16: np.ndarray) -> np.ndarray:
    return (audio_i16.astype(np.float32) / 32767.0).astype(np.float32)

def webrtc_process_block(ap: AudioProcessor, block_f32: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    WebRTC AudioProcessor operates on 10ms frames = 160 samples @ 16kHz (320 bytes int16).
    We receive 100ms blocks, so we split into 10ms chunks, process each, and stitch back.
    Returns:
      clean_block_f32, any_speech_vad
    """
    i16 = float32_to_int16(block_f32)
    n = i16.size

    frame_len = 160  # 10ms @ 16k
    # pad to multiple of 160
    pad = (-n) % frame_len
    if pad:
        i16 = np.pad(i16, (0, pad))
        n = i16.size

    out_frames = []
    any_speech = False

    for start in range(0, n, frame_len):
        frame = i16[start:start + frame_len]
        frame_bytes = frame.tobytes()  # 320 bytes

        result = ap.Process10ms(frame_bytes)
        any_speech = any_speech or bool(result.is_speech)

        # result.audio is bytes (int16 little-endian)
        out_i16 = np.frombuffer(result.audio, dtype=np.int16)
        out_frames.append(out_i16)

    out_i16_all = np.concatenate(out_frames)
    if pad:
        out_i16_all = out_i16_all[:out_i16_all.size - pad]

    return int16_to_float32(out_i16_all), any_speech

def main():
    if RECORD_SESSION_AUDIO:
        os.makedirs(SESSION_OUT_DIR, exist_ok=True)

    print(f"Loading Whisper model: {WHISPER_MODEL} ...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    print("\nWhisper STT test + WebRTC Noise Suppression. Ctrl+C to quit.\n")
    print(f"[NS] noise_suppression_level={NOISE_SUPPRESSION_LEVEL} (0..4), auto_gain_dbfs={AUTO_GAIN_DBFS} (0..31, 0=off)\n")

    # WebRTC audio processor (16 kHz mono int16 frames only) :contentReference[oaicite:2]{index=2}
    ap = AudioProcessor(AUTO_GAIN_DBFS, NOISE_SUPPRESSION_LEVEL)

    speaking = False
    buf_clean: list[np.ndarray] = []
    last_voice_time = None

    all_audio_raw: list[np.ndarray] = []
    all_audio_clean: list[np.ndarray] = []

    min_samples = int(MIN_UTTERANCE_SEC * SAMPLE_RATE)

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
            block_raw = audio_q.get()

            if RECORD_SESSION_AUDIO:
                all_audio_raw.append(block_raw)

            # --- WebRTC NS/AGC ---
            block_clean, vad_speech = webrtc_process_block(ap, block_raw)

            if RECORD_SESSION_AUDIO and SAVE_RAW_AND_CLEAN:
                all_audio_clean.append(block_clean)

            # --- VAD / segmentation ---
            # Use cleaned audio for RMS (more stable), and optionally WebRTC VAD as extra signal.
            level = rms(block_clean)
            now = time.time()

            voice = (level > RMS_THRESHOLD) or vad_speech

            if voice:
                if not speaking:
                    speaking = True
                    buf_clean = []
                    print("[VAD] speech start")
                buf_clean.append(block_clean)
                last_voice_time = now
            else:
                if speaking:
                    # keep a bit of trailing silence in the buffer
                    buf_clean.append(block_clean)

                    if last_voice_time is not None and (now - last_voice_time) >= END_SILENCE_SEC:
                        speaking = False
                        audio_clean = np.concatenate(buf_clean) if buf_clean else np.array([], dtype=np.float32)

                        if audio_clean.size < min_samples:
                            print("[VAD] ignored (too short)\n")
                            continue

                        print("[VAD] speech end -> transcribing...")

                        try:
                            segments, _ = model.transcribe(
                                audio_clean,
                                language=LANGUAGE,
                                beam_size=1,
                                vad_filter=False,  # we already segmented
                            )
                            text = " ".join(s.text for s in segments).strip()
                        except Exception as e:
                            print(f"[STT ERROR] {e}\n")
                            continue

                        print(f"[STT] {text}\n" if text else "[STT] (empty)\n")

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")

    finally:
        try:
            if stream.active:
                stream.stop()
        except Exception:
            pass

        if RECORD_SESSION_AUDIO and all_audio_raw:
            ts = datetime.now().isoformat(timespec="seconds").replace(":", "-")

            raw = np.concatenate(all_audio_raw)
            raw_path = os.path.join(SESSION_OUT_DIR, f"full_session_RAW_{ts}.wav")
            save_wav_float32_mono(raw_path, raw, SAMPLE_RATE)
            print(f"[REC] saved RAW session:   {raw_path}")

            if SAVE_RAW_AND_CLEAN:
                clean = np.concatenate(all_audio_clean) if all_audio_clean else raw
                clean_path = os.path.join(SESSION_OUT_DIR, f"full_session_CLEAN_{ts}.wav")
                save_wav_float32_mono(clean_path, clean, SAMPLE_RATE)
                print(f"[REC] saved CLEAN session: {clean_path}")

        elif RECORD_SESSION_AUDIO:
            print("[REC] No audio captured; nothing to save.")

if __name__ == "__main__":
    main()
