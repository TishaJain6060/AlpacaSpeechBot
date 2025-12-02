# stt_module.py

import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from config import SAMPLE_RATE, CHUNK_SIZE, RMS_THRESHOLD, WHISPER_MODEL_NAME

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
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio Status: {status}")

        audio_data = np.squeeze(indata.copy())
        rms = np.sqrt(np.mean(audio_data**2))

        # Push only meaningful audio
        if rms > self.rms_threshold:
            audio_queue.put(audio_data)

    def start_stream(self):
        print(f"Starting audio stream at {self.samplerate}Hz...")
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
# SpeechToText (Live Transcription)
# =====================================

class SpeechToText:
    """Handles live transcription using faster-whisper."""
    def __init__(self, model_name=WHISPER_MODEL_NAME):
        print(f"🔧 Loading faster-whisper model ({model_name})...")
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe_audio_chunk(self, audio_data: np.ndarray) -> str:
        """Transcribes a single chunk of audio."""
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=5,
            language=None,
            vad_filter=True
        )
        return " ".join(segment.text for segment in segments).strip()

    def run_transcription_loop(self, pipeline_handler, exit_flag):
        """Continuously processes audio chunks as they arrive."""
        print("Transcription")
        while not exit_flag.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text = self.transcribe_audio_chunk(audio_chunk)

            if text:
                print(f"\n You said: {text}")
                if pipeline_handler:
                    pipeline_handler.process_stt_result(text)
