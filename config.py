# config.py

import warnings

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# --- Audio Configuration ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 20      # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
RMS_THRESHOLD = 0.015       # Volume threshold for noise gate

# --- Model Configuration ---
WHISPER_MODEL_NAME = "tiny"
PIPER_VOICE_FILE = "en_US-lessac-medium.onnx"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
# IMPORTANT: Use environment variables for API keys in a real project!
GEMINI_API_KEY = "AIzaSyAwRx55yf-VQ1I4ycZT6dgxCe26dREuOzI"