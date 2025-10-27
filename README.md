Gemini Voice Assistant with Piper TTS

A local voice assistant that listens to your speech, queries Google Gemini for responses, and speaks them using Piper TTS. Interaction is intentended for integration with AgileX robot for an interactive robot.

Features

Live speech recognition using speech_recognition

Google Gemini integration via google-generativeai

High-quality local text-to-speech using Piper TTS

Fully dynamic: multiple interactions, no pyttsx3 issues

Exit program with trigger phrase "thank you"

Dependencies
Python

Python 3.10+ recommended

Python Packages

Install via pip:

pip install speechrecognition
pip install google-generativeai
pip install piper-tts
pip install sounddevice
pip install numpy
pip install onnxruntime


⚠️ For GPU acceleration of Piper TTS (optional):

pip install onnxruntime-gpu

OS-Specific Notes
Windows

Ensure you have Visual C++ Redistributable installed (required by onnxruntime)

Use a microphone compatible with speech_recognition

sounddevice may require portaudio which is bundled in pip wheels

macOS

sounddevice requires portaudio:

brew install portaudio


Use the built-in microphone or an external USB mic

Setup

Clone or download this repository

Download a Piper voice

Example using Piper CLI:

python -m piper.download_voices en_US-lessac-medium


This will download the .onnx file (e.g., en_US-lessac-medium.onnx) into your directory.

Update voice path in the script

voice_path = "en_US-lessac-medium.onnx"


Set your Google Gemini API key

GEMINI_API_KEY = "YOUR_KEY_HERE"

Usage

Run the script:

python speech_to_text.py


Speak into your microphone

Gemini will respond to your input

The response will be spoken out loud using Piper TTS

Say "thank you" to exit the program

Configuration
Piper TTS Synthesis Parameters
syn_config = SynthesisConfig(
    volume=0.8,          # Voice volume (0-1)
    length_scale=1.0,    # Speed modifier (>1 = slower)
    noise_scale=0.6,     # Controls audio variation
    noise_w_scale=0.6,   # Controls prosody variation
    normalize_audio=True # Keep waveform normalized
)

Gemini Model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

Notes

Works offline for TTS; online only for Gemini API calls

The first interaction may have a small delay due to microphone calibration and TTS initialization

Compatible with multiple Piper voices — simply replace .onnx file

If using requirements.txt 
Save requirements.txt in the same directory as your script.

Install all dependencies:

pip install -r requirements.txt


⚠️ On macOS, make sure PortAudio is installed for sounddevice:

brew install portaudio


Optional for GPU:

pip install onnxruntime-gpu