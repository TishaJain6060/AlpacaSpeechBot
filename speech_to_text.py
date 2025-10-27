import speech_recognition as sr
import google.generativeai as genai
from piper import PiperVoice, SynthesisConfig
import sounddevice as sd
import numpy as np
import wave
import time

# ========== CONFIG ==========
GEMINI_API_KEY = "YourKey"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ========== LOAD TTS VOICE ==========
voice_path = "en_US-lessac-medium.onnx"  # path to downloaded Piper voice
voice = PiperVoice.load(voice_path, use_cuda=False)  # Set True if you have GPU & onnxruntime-gpu

# Optional: customize synthesis parameters
syn_config = SynthesisConfig(
    volume=0.8,
    length_scale=1.0,
    noise_scale=0.6,
    noise_w_scale=0.6,
    normalize_audio=True
)

# ========== FUNCTIONS ==========
def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini Error: {e}"

def speak_text(text):
    # Synthesizes audio in memory
    audio_chunks = voice.synthesize(text, syn_config=syn_config)
    for chunk in audio_chunks:
        # chunk.audio_int16_bytes is the raw PCM data
        audio_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32)
        audio_data /= 32768.0  # scale int16 to float32
        sd.play(audio_data, samplerate=chunk.sample_rate)
        sd.wait()  # wait for chunk to finish

# ========== MAIN LOOP ==========
recognizer = sr.Recognizer()

def main():
    print("\n🎤 Speak now! Say 'thank you' to exit.\n")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while True:
            try:
                print("🗣️ Listening...")
                audio = recognizer.listen(source, phrase_time_limit=10)
                user_text = recognizer.recognize_google(audio).strip()
                print(f"\n🗣️ You said: {user_text}")

                if "thank you" in user_text.lower():
                    print("\n👋 Exiting program. Goodbye!")
                    speak_text("Goodbye!")
                    break

                # Ask Gemini
                print("🤖 Thinking...")
                reply = ask_gemini(user_text)
                print(f"💬 Gemini: {reply}\n")

                # Speak Gemini reply
                speak_text(reply)

            except sr.UnknownValueError:
                print("❌ Didn't catch that. Try again.")
            except sr.RequestError as e:
                print(f"⚠️ Could not request results; {e}")
            except KeyboardInterrupt:
                print("\n🛑 Interrupted manually. Exiting.")
                break
            time.sleep(0.5)

if __name__ == "__main__":
    main()


