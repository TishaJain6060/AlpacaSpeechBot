# stt_module.py

import queue
import sounddevice as sd
import numpy as np
import threading
from faster_whisper import WhisperModel
from config import SAMPLE_RATE, CHUNK_SIZE, RMS_THRESHOLD, WHISPER_MODEL_NAME

# For standalone testing metrics:
import time
import jiwer
from scipy.io import wavfile
import os
import random

# Global Queue for inter-thread communication
audio_queue = queue.Queue()

# =====================================
# AudioCapture 
# =====================================

class AudioCapture:
    """Manages audio input stream and basic noise filtering."""
    def __init__(self, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, rms_threshold=RMS_THRESHOLD):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.rms_threshold = rms_threshold

    def audio_callback(self, indata, frames, time_info, status):
        # ... (implementation remains the same)
        if status:
            print(f"Audio Status: {status}")
        audio_data = np.squeeze(indata.copy())
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > self.rms_threshold:
            audio_queue.put(audio_data)

    def start_stream(self):
        # ... (implementation remains the same)
        print(f"🎙️ Starting audio stream at {self.samplerate}Hz...")
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype='float32',
            callback=self.audio_callback,
            blocksize=self.blocksize,
        )
        self.stream.start()
        return self.stream 


# =====================================
# SpeechToText
# =====================================

class SpeechToText:
    """Handles transcription using faster-whisper."""
    def __init__(self, model_name=WHISPER_MODEL_NAME):
        print(f"🔧 Loading faster-whisper model ({model_name})...")
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8") 

    def transcribe_audio_chunk(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a single chunk of audio data.
        """
        segments, info = self.model.transcribe(
            audio_data, 
            beam_size=5, 
            language=None, 
            vad_filter=True
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text

    def run_transcription_loop(self, pipeline_handler, exit_flag):
        # ... (main pipeline loop logic remains the same)
        print("🎧 Transcription process started.")
        while not exit_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.1) 
            except queue.Empty:
                continue
            text = self.transcribe_audio_chunk(audio_chunk)
            if text:
                print(f"\n🗣️ You said: {text}")
                if pipeline_handler:
                    pipeline_handler.process_stt_result(text)

# =====================================
# STANDALONE TEST SCRIPT (Run only when executed directly)
# =====================================

def _load_audio_file(file_path):
    """Loads a WAV file and returns it as a float32 numpy array."""
    try:
        samplerate, data = wavfile.read(file_path)
        if data.dtype == np.int16:
            audio_data = data.astype(np.float32) / 32768.0
        else:
             audio_data = data.astype(np.float32)
             
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        return audio_data, samplerate
    except Exception as e:
        print(f"❌ Error loading audio file: {e}")
        return None, None

def main_test_stt():
    """Function to run the STT latency and accuracy test with percentiles."""
    
    # --- CONFIGURATION ---
    # 1. Provide a test audio file path
    TEST_AUDIO_FILE = "testingaudio.wav" 
    # 2. Provide the exact text spoken in that file (ground truth)
    GROUND_TRUTH = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired" 
    # 3. Number of times to run the test for statistical accuracy
    NUM_RUNS = 20 
    # ---------------------

    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"❌ Test audio file not found: {TEST_AUDIO_FILE}")
        return

    print(f"--- STT Test ({NUM_RUNS} Runs) using {WHISPER_MODEL_NAME.upper()} ---")
    stt_engine = SpeechToText()
    audio_data, samplerate = _load_audio_file(TEST_AUDIO_FILE)
    
    if audio_data is None: return

    audio_duration = len(audio_data) / samplerate
    print(f"File Duration: {audio_duration:.2f}s | Ground Truth: '{GROUND_TRUTH}'")
    
    # --- Performance Collection ---
    latencies = []
    wer_scores = []
    
    for i in range(1, NUM_RUNS + 1):
        # 1. Latency Measurement
        start_time = time.perf_counter()
        
        # Call the isolated transcription method
        hypothesis = stt_engine.transcribe_audio_chunk(audio_data)
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)
        
        # 2. Accuracy Measurement
        wer_score = jiwer.wer([GROUND_TRUTH], [hypothesis])
        wer_scores.append(wer_score)
        
        print(f"Run {i}/{NUM_RUNS}: Latency={latency:.3f}s, WER={wer_score*100:.1f}%")

    # --- Final Statistical Analysis ---
    
    # Latency Percentiles
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Average Metrics
    avg_latency = np.mean(latencies)
    avg_wer = np.mean(wer_scores)
    
    # 4. Display Results
    print("\n" + "="*50)
    print("✅ STT PERFORMANCE BENCHMARK RESULTS")
    print(f"MODEL: {WHISPER_MODEL_NAME.upper()} | RUNS: {NUM_RUNS} | AUDIO DURATION: {audio_duration:.2f}s")
    print("="*50)

    ## --- Accuracy Metrics ---
    print("🤖 ACCURACY (Word Error Rate):")
    print(f"  > Average WER: {avg_wer * 100:.2f}%")
    print(f"  > Average Accuracy: {(1 - avg_wer) * 100:.2f}%")
    
    ## --- Latency Metrics ---
    print("\n⏱️ LATENCY (Total Transcription Time):")
    print(f"  > P50 (Median Latency): {p50_latency:.3f} seconds")
    print(f"  > P95 Latency: {p95_latency:.3f} seconds")
    print(f"  > P99 Latency: {p99_latency:.3f} seconds")

    ## --- Real-Time Factor ---
    print("\n⚡ REAL TIME FACTOR (RTF):")
    print(f"  > Average RTF: {avg_latency / audio_duration:.2f} (Target < 0.5)")
    print(f"  > P95 RTF: {p95_latency / audio_duration:.2f}")
    print("="*50)


if __name__ == "__main__":
    main_test_stt()

