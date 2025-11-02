import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import time
import warnings
import wave
import re
from google import genai
from google.genai.types import GenerateContentConfig
from piper import PiperVoice

# =====================================
# ⚙️ CONFIGURATION
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
GEMINI_API_KEY = "AIzaSyAwRx55yf-VQ1I4ycZT6dgxCe26dREuOzI" # IMPORTANT: Use environment variables in production!

# Global Queue for inter-thread communication
audio_queue = queue.Queue()

# =====================================
# 🎙️ 1. AudioCapture Module
# =====================================
class AudioCapture:
    """Manages audio input stream and basic noise filtering."""
    def __init__(self, samplerate, blocksize, rms_threshold):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.rms_threshold = rms_threshold

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice to push audio to the queue."""
        if status:
            print(f"Audio Status: {status}")
        
        # Simple Noise Gate/Silence Detection using RMS
        audio_data = np.squeeze(indata.copy())
        # Calculate Root Mean Square (RMS) volume
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > self.rms_threshold:
            audio_queue.put(audio_data)
        # else: print(".", end="", flush=True) # Optional: uncomment to see when audio is ignored

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
# 👂 2. SpeechToText Module (Faster Whisper)
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
            language=None, # Auto-detect language
            vad_filter=True # Use Voice Activity Detection for better segmenting
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text

    def run_transcription_loop(self, pipeline_handler):
        """Continuously pulls audio and transcribes it."""
        print("🎧 Transcription process started.")
        while True:
            # Block until an audio chunk is available
            audio_chunk = audio_queue.get() 
            
            # Transcription can be slow, run in a separate thread if needed, 
            # but here it processes sequentially to clear the queue.
            text = self.transcribe(audio_chunk)

            if text:
                print(f"\n🗣️ You said: {text}")
                # Pass the transcribed text to the next stage of the pipeline
                threading.Thread(
                    target=pipeline_handler.handle, 
                    args=(text,), 
                    daemon=True
                ).start()

# =====================================
# 🧠 3. LLMClient Module (Gemini)
# =====================================
class LLMClient:
    """Handles communication with the Gemini API."""
    def __init__(self, api_key, model_name):
        print("🔧 Initializing Gemini client...")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """Generates a response from Gemini."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
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
# 🗣️ 4. TextToSpeech Module (Piper)
# =====================================
class TextToSpeech:
    """Handles text-to-speech synthesis using Piper."""
    def __init__(self, voice_file):
        print("🔧 Loading Piper voice...")
        self.voice = PiperVoice.load(voice_file)

    def _clean_text(self, text: str) -> str:
        """Removes markdown and redundant characters before TTS."""
        import re
        
        # 1. Remove Markdown Bolding/Italics (* and _)
        # This covers cases like **word** or *word*
        text = text.replace('*', '').replace('_', '')
        
        # 2. Remove Redundant/Specific Punctuation
        # Remove slashes, pipes, brackets, and similar non-verbal characters
        # Commas (,), periods (.), question marks (?), and exclamation points (!) 
        # are generally kept as they affect natural pausing/intonation.
        text = re.sub(r'[\\/|\[\]{}()]', '', text)
        
        # 3. Handle excessive whitespace created by removal
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def speak(self, text: str):
        """Synthesizes text and plays the audio."""
        
        # NEW: Clean the text before speaking
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text:
            print("🔊 TTS skipped: Cleaned text was empty.")
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
# 🚀 5. Generizable Pipeline Handler
# =====================================
class PipelineHandler:
    """Coordinates the LLM and TTS steps."""
    def __init__(self, llm_client: LLMClient, tts_engine: TextToSpeech):
        self.llm_client = llm_client
        self.tts_engine = tts_engine

    def handle(self, user_text: str):
        """Run Gemini + TTS asynchronously for each detected phrase."""
        start = time.time()
        
        # 1. LLM Generation
        response = self.llm_client.generate_response(user_text)
        mid = time.time()
        
        # 2. TTS Synthesis and Playback
        self.tts_engine.speak(response)
        end = time.time()
        
        print(f"⏱️ Latency | Gemini: {mid - start:.2f}s | TTS: {end - mid:.2f}s")

# =====================================
# 🧪 MAIN EXECUTION
# =====================================
def main():
    # --- Initialization ---
    tts_engine = TextToSpeech(PIPER_VOICE_FILE)
    llm_client = LLMClient(GEMINI_API_KEY, GEMINI_MODEL_NAME)
    stt_engine = SpeechToText(WHISPER_MODEL_NAME)
    
    pipeline_handler = PipelineHandler(llm_client, tts_engine)
    
    audio_capture = AudioCapture(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        rms_threshold=RMS_THRESHOLD
    )
    
    # --- Start Processes ---
    # 1. Start the audio input stream
    audio_capture.start_stream() 
    
    # 2. Start the transcription loop in a dedicated thread
    whisper_thread = threading.Thread(
        target=stt_engine.run_transcription_loop, 
        args=(pipeline_handler,), 
        daemon=True
    )
    whisper_thread.start()
    
    print("\n🎧 Listening... Speak into your mic (Ctrl+C to stop).")
    
    # --- Keep Main Thread Alive ---
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping voice assistant. Goodbye!")
    finally:
        # Gracefully stop the stream
        if audio_capture.stream.active:
            audio_capture.stream.stop()
        
if __name__ == "__main__":
    main()