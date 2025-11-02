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
GEMINI_API_KEY = "YourKey" 

# Global Queue & State
audio_queue = queue.Queue()
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
        # else: print(".", end="", flush=True) # (optional) uncomment to see when audio is ignored

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
            language=None, # Auto-detect language
            vad_filter=True # Use Voice Activity Detection for better segmenting
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text

    def run_transcription_loop(self, pipeline_handler):
        """Continuously pulls audio and transcribes it."""
        global confirming_exit

        print("🎧 Transcription process started.")
        while not exit_flag.is_set():
            # Use a timeout so the loop can check the exit_flag periodically
            try:
                audio_chunk = audio_queue.get(timeout=0.1) 
            except queue.Empty:
                continue

            text = self.transcribe(audio_chunk)

            if text:
                print(f"\n🗣️ You said: {text}")
                
                # --- Exit Check Logic ---
                if confirming_exit:
                    # If we are confirming exit, handle 'yes' or 'no' response
                    pipeline_handler.handle_exit_confirmation(text)
                else:
                    # Otherwise, check for a farewell phrase
                    if pipeline_handler.is_farewell(text):
                        confirming_exit = True
                        pipeline_handler.handle_farewell()
                    else:
                        # Continue normal conversation flow
                        threading.Thread(
                            target=pipeline_handler.handle_conversation, 
                            args=(text,), 
                            daemon=True
                        ).start()

# =====================================
# 3. LLMClient Module (Gemini)
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
# 4. TextToSpeech Module (Piper)
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
# 5. Generizable Pipeline Handler
# =====================================
class PipelineHandler:
    """Coordinates the LLM and TTS steps and manages exit."""
    def __init__(self, llm_client: LLMClient, tts_engine: TextToSpeech):
        self.llm_client = llm_client
        self.tts_engine = tts_engine

        # Regex to detect common farewell phrases, case-insensitive
        self.farewell_pattern = re.compile(r'\b(thank\s?you|thanks|cheers|bye|goodbye|that\'s\s?it)\b', re.IGNORECASE)
        self.affirmative_pattern = re.compile(r'\b(yes|yeah|yep|sure|continue)\b', re.IGNORECASE)
        self.negative_pattern = re.compile(r'\b(no|nope|nah|exit|stop)\b', re.IGNORECASE)
    
    def is_farewell(self, text: str) -> bool:
        """Checks if the user's input is a farewell phrase."""
        return bool(self.farewell_pattern.search(text))

    def handle_conversation(self, user_text: str): 
        """Run Gemini + TTS asynchronously for a normal conversation turn."""
        start = time.time()
        
        # 1. LLM Generation
        response = self.llm_client.generate_response(user_text)
        mid = time.time()
        
        # 2. TTS Synthesis and Playback
        self.tts_engine.speak(response)
        end = time.time()
        
        print(f"⏱️ Latency | Gemini: {mid - start:.2f}s | TTS: {end - mid:.2f}s")
        
    def handle_farewell(self):
        """Initiates the exit confirmation sequence."""
        global confirming_exit
        
        confirmation_text = "Of course! Happy to help. Is there anything else I can assist you with?"
        self.tts_engine.speak(confirmation_text)
        # Note: confirming_exit is set to True in the STT loop caller

    def handle_exit_confirmation(self, user_text: str):
        """Handles the user's response to the 'Is there anything else?' prompt."""
        global confirming_exit
        
        if self.negative_pattern.search(user_text):
            # User confirms exit (e.g., "no", "nope", "that's it")
            self.tts_engine.speak("You're welcome! Goodbye.")
            exit_flag.set() # Set the flag to terminate the program
        elif self.affirmative_pattern.search(user_text):
            # User wants to continue (e.g., "yes", "yeah", "I have one more thing")
            confirming_exit = False # Reset flag to continue normal conversation
            self.tts_engine.speak("Great! What else can I help you with?")
        else:
            # Ambiguous response, ask again
            self.tts_engine.speak("I'm sorry, I didn't catch that. Do you need further assistance? Say 'yes' or 'no'.")

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
    
    print("\n🎧 Listening... Speak into your mic (Ctrl+C OR say 'thank you' to stop).")
    
    # --- Keep Main Thread Alive (Monitors exit_flag) ---
    try:
        # Wait until the exit_flag is set (by handle_exit_confirmation)
        exit_flag.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by KeyboardInterrupt.")
    finally:
        # Stop the stream
        if audio_capture.stream.active:
            audio_capture.stream.stop()
        print("\n⏹️ Voice assistant shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()