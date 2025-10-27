
# Description

This program converts speech to text using OpenAI Whisper, generates responses using Google Gemini, and speaks responses with Piper TTS. Interaction is intentended for integration with AgileX robot for an interactive robot.

# Requirements

Python 3.10+

sounddevice (for audio input/output)

numpy

whisper (OpenAI Whisper)

piper-tts (Piper TTS)

google-genai (Gemini API)

# Installation
1️. Clone this repository
git clone <repo-url>
cd gemini_voice_assistant

2️. Install dependencies

Create a requirements.txt:

sounddevice
numpy
whisper
piper-tts
google-genai


Then install:

# Linux
python3 -m pip install -r requirements.txt

# Windows
python -m pip install -r requirements.txt

3️. Download a Piper voice
python -m piper.download_voices en_US-lessac-medium


This will generate a .onnx file (e.g., en_US-lessac-medium.onnx) that you will use in the script.

4️. Set Google Gemini API Key

Linux/macOS:

export GEMINI_API_KEY="your_api_key_here"


Windows (PowerShell):

setx GEMINI_API_KEY "your_api_key_here"

Usage

Run the voice assistant:

python speech_to_text.py


Speak into your microphone.

Say "thank you" to exit the program.

The pipeline will:

Transcribe your speech using Whisper.

Generate a response using Gemini LLM.

Speak the response using Piper TTS.

Notes

Whisper does not require an API key.

Gemini requires a valid API key; ensure it’s set in your environment.

Piper can optionally use GPU if you have onnxruntime-gpu installed.

You can adjust volume, speed, and other TTS settings in the script via SynthesisConfig.

# Directory Structure
gemini_voice_assistant/
│
├── speech_to_text.py      # main modular voice assistant script
├── en_US-lessac-medium.onnx  # Piper voice file
├── requirements.txt
└── README.md

# Supported Platforms
Platform	Notes
Windows 10+	Works with default Python audio drivers.
Linux (Ubuntu/Debian)	May require libasound2-dev for sounddevice.
Troubleshooting

Whisper errors: Ensure your microphone is working and the sampling rate is set to 16kHz.

Gemini errors: Verify your API key is correct and the model exists (gemini-2.0-flash or latest).

Piper errors: Ensure the .onnx voice file exists and piper-tts is installed.

Latency: For benchmarking, enable print statements in STT/LLM/TTS functions.