# tts_module.py

import wave
import numpy as np
import sounddevice as sd
import re
import time
from piper import PiperVoice
from config import PIPER_VOICE_FILE
import os
import numpy as np

# =====================================
# CORE CLASS: TextToSpeech
# =====================================

class TextToSpeech:
    """Handles text-to-speech synthesis using Piper."""
    def __init__(self, voice_file=PIPER_VOICE_FILE):
        print("🔧 Loading Piper voice...")
        # NOTE: This line might take a moment to download the voice file on first run.
        self.voice = PiperVoice.load(voice_file)

    def _clean_text(self, text: str) -> str:
        """Removes markdown and redundant characters before TTS."""
        text = text.replace('*', '').replace('_', '')
        text = re.sub(r'[\\/|\[\]{}()]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def synthesize_to_wav(self, text: str, output_path: str) -> tuple[float, float, str]:
        """
        Synthesizes text to a WAV file and measures synthesis latency.
        
        Returns: (synthesis_latency, audio_duration_s, cleaned_text)
        """
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return 0.0, 0.0, ""

        # --- SYNTHESIS MEASUREMENT ---
        start_time = time.perf_counter()
        
        try:
            with wave.open(output_path, "wb") as wav_file:
                # Piper performs the synthesis into the wav_file stream
                self.voice.synthesize_wav(cleaned_text, wav_file)
            
            end_time = time.perf_counter()
            synthesis_latency = end_time - start_time
            
            # Read file length to get Audio Duration
            with wave.open(output_path, "rb") as wf:
                audio_duration_s = wf.getnframes() / wf.getframerate()
            
            return synthesis_latency, audio_duration_s, cleaned_text
        
        except Exception as e:
            print(f"❌ Piper synthesis error: {e}")
            return 0.0, 0.0, ""

    def speak(self, text: str):
        """Synthesizes text and plays the audio (for pipeline use)."""
        output_file = "temp_reply.wav"
        
        synthesis_latency, audio_duration_s, cleaned_text = self.synthesize_to_wav(text, output_file)
        
        if synthesis_latency > 0.0:
             print(f"🔊 Speaking text: {cleaned_text}")
             try:
                 # Read and Play the WAV file
                 with wave.open(output_file, "rb") as wf:
                     data = wf.readframes(wf.getnframes())
                     audio = np.frombuffer(data, dtype=np.int16)
                     sd.play(audio, wf.getframerate())
                     sd.wait() 
                 os.remove(output_file) # Clean up temp file
             except Exception as e:
                 print(f"❌ Playback error: {e}")

        return synthesis_latency # In pipeline, we return synthesis latency
        

# =====================================
# STANDALONE TEST SCRIPT (Latency & RTF)
# =====================================

def main_test_tts():
    """Function to run the TTS latency and RTF test with percentiles."""
    
    # --- CONFIGURATION ---
    # The text to synthesize and test latency on. Should be a typical response length (e.g., 20-40 words).
    TEST_PHRASE = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired."
    NUM_RUNS = 20 # Run enough times for statistical stability
    # ---------------------

    print(f"--- TTS Latency Test ({NUM_RUNS} Runs) ---")
    tts_engine = TextToSpeech()
    
    latencies = []
    audio_durations = []
    temp_file = "tts_test_output.wav"
    
    # Pre-run to get the target duration and prime the system
    _, audio_duration, _ = tts_engine.synthesize_to_wav(TEST_PHRASE, temp_file)
    if audio_duration == 0.0:
        print("❌ Pre-run failed. Check your Piper setup.")
        return
    
    print(f"Test Phrase: '{TEST_PHRASE}'")
    print(f"Target Audio Duration: {audio_duration:.2f}s")
    
    # --- Performance Collection ---
    for i in range(1, NUM_RUNS + 1):
        # We only measure Synthesis time, as playback is determined by audio duration.
        synthesis_latency, duration, _ = tts_engine.synthesize_to_wav(TEST_PHRASE, temp_file)
        latencies.append(synthesis_latency)
        audio_durations.append(duration) # Should be constant
        
        rtf = synthesis_latency / duration if duration > 0 else 0
        print(f"Run {i}/{NUM_RUNS}: Synth Latency={synthesis_latency:.3f}s, RTF={rtf:.2f}")

    # Clean up the test file
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # --- Final Statistical Analysis ---
    
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    avg_rtf = avg_latency / audio_duration
    p95_rtf = p95_latency / audio_duration
    
    # 4. Display Results
    print("\n" + "="*50)
    print("✅ TTS PERFORMANCE BENCHMARK RESULTS")
    print(f"MODEL: Piper | RUNS: {NUM_RUNS} | TEST TEXT LENGTH: {len(TEST_PHRASE)} chars")
    print("="*50)
    
    # TTS "Accuracy" (RTF)
    print("⚡ SYNTHESIS SPEED (Real-Time Factor - RTF):")
    print(f" > Average RTF: {avg_rtf:.2f} (Target < 0.5 is Excellent)")
    print(f" > P95 RTF: {p95_rtf:.2f}")

    # Latency Metrics
    print("\n⏱️ SYNTHESIS LATENCY (Time to Generate Audio):")
    print(f" > P50 (Median Latency): {p50_latency:.3f} seconds")
    print(f" > P95 Latency: {p95_latency:.3f} seconds")
    print(f" > P99 Latency: {p99_latency:.3f} seconds")

    print("\n💡 NOTE: Total user perceived latency will be Synthesis Time + Playback Time.")
    print("="*50)

if __name__ == "__main__":
    main_test_tts()