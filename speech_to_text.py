import sounddevice as sd
import numpy as np
import tempfile
import wave
from openai import OpenAI

client = OpenAI(api_key="APIKEYHERE")  

# -------- RECORD SPEECH --------
def record_audio(duration=5, samplerate=44100):
    print("🎤 Speak now...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")
    return recording, samplerate

# -------- SAVE TEMP WAV --------
def save_temp_wav(audio_data, samplerate):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return tmp.name

# -------- TRANSCRIBE --------
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        # transcription = client.audio.transcriptions.create(
        #     model="gpt-4o-mini-transcribe",  # or "whisper-1"
        #     file=f
        # )
        transcription = client.audio.transcriptions.create(
    model="whisper-1",  # cheaper and less quota-heavy
    file=f
    )
    return transcription.text

# -------- GET GPT RESPONSE --------
def get_gpt_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# -------- MAIN LOOP --------
if __name__ == "__main__":
    audio, sr = record_audio(duration=5)
    wav_path = save_temp_wav(audio, sr)

    text = transcribe_audio(wav_path)
    print(f"\n🗣️ You said: {text}")

    reply = get_gpt_response(text)
    print(f"\n🤖 GPT says: {reply}\n")
