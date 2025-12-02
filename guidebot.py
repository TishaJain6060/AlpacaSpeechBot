# #!/usr/bin/env python3
# import os
# import sys
# import time
# import json
# import speech_recognition as sr
# import re
# import google.generativeai as genai


# def state(msg):
#     print(f"\n>>> {msg}\n")
#     sys.stdout.flush()


# # -----------------------------------------------------
# # CONFIG
# # -----------------------------------------------------
# GENAI_API_KEY = "AIzaSyAwRx55yf-VQ1I4ycZT6dgxCe26dREuOzI"
# genai.configure(api_key=GENAI_API_KEY)

# WAKE_PHRASES = ["hey alpaca", "can you help me", "hello guide", "alpaca"]
# LANGUAGE = "en-US"
# COMMAND_TIMEOUT = 12
# CONVO_INACTIVITY = 15     # ← stay active for 15 seconds

# END_PHRASES = [
#     "thank you", "that's all", "bye", "stop", "exit", "that is all", "thanks"
# ]

# memory = {
#     "current_position": 0    # treat this as "latest room"
# }

# # -----------------------------------------------------
# # Room coordinates mapping
# # -----------------------------------------------------
# ROOM_COORDS = {
#     "3140": [0.0, 17.706],
#     "3141": [0.0, 17.706],
#     "3150": [0.0, 8.743],
#     "3160": [3.2, 7.1],
#     "3161": [4.567, -4.880],
#     "3170": [0.0, -9.543],
#     "3171": [0.0, -9.543],
#     # Add more rooms here...
# }


# # -----------------------------------------------------
# # LLM extraction
# # -----------------------------------------------------
# EXTRACTION_PROMPT = """
# You are a navigation interpreter. ONLY output JSON.

# Task:
# - Identify the target room number.
# - Identify source room ONLY if user explicitly says it.
# - Convert number-words to digits.
# - Ignore irrelevant text.

# Output EXACTLY this format:
# {{
#   "source_room": <int or null>,
#   "target_room": <int or null>,
#   "notes": "<short reasoning>"
# }}

# User said: "{utterance}"
# """


# def clean_json_block(text):
#     """Remove markdown fences if Gemini returns them."""
#     text = text.strip()
#     if text.startswith("```"):
#         text = re.sub(r"```.*?\n", "", text, flags=re.DOTALL)
#         text = text.replace("```", "")
#     return text.strip()


# def ask_llm_for_rooms(utterance: str):
#     """Robust JSON extractor for Gemini responses."""
#     prompt = EXTRACTION_PROMPT.format(utterance=utterance)
#     model = genai.GenerativeModel("gemini-2.5-flash")

#     try:
#         response = model.generate_content(prompt)
#         raw = response.text

#         # --- Clean markdown fences ---
#         raw = raw.replace("```json", "").replace("```", "").strip()

#         # --- Extract JSON block using regex ---
#         match = re.search(r"\{[\s\S]*\}", raw)
#         if match:
#             json_text = match.group(0)
#             return json.loads(json_text)

#         print("[LLM ERROR] Could not locate JSON block.")
#         return {"source_room": None, "target_room": None, "notes": "JSON not found"}

#     except Exception as e:
#         print("[LLM ERROR]", e)
#         return {"source_room": None, "target_room": None, "notes": "exception"}


# # -----------------------------------------------------
# # Speech helpers
# # -----------------------------------------------------
# def passive_listen(recognizer, mic):
#     """Wait for wake word."""
#     state("Passive Listening — waiting for wake phrase...")
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source, duration=0.3)
#         audio = recognizer.listen(source, phrase_time_limit=4)

#     try:
#         heard = recognizer.recognize_google(audio, language=LANGUAGE).lower()
#         print(f"[hearing] {heard}")
#         return any(w in heard for w in WAKE_PHRASES)
#     except:
#         return False


# def capture_sentence(recognizer, mic, timeout=12):
#     state(f"[listening…] speak now (max {timeout}s)")
#     with mic as source:
#         audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)

#     state("[processing…]")
#     try:
#         text = recognizer.recognize_google(audio, language=LANGUAGE)
#         state(f"Captured!: {text}")
#         return text.strip()
#     except:
#         state("Could not understand.")
#         return ""


# # -----------------------------------------------------
# # Helper: Detect if user wants navigation
# # -----------------------------------------------------
# NAV_KEYWORDS = ["room", "take me", "go to", "where is", "directions", "navigate", "get to", "get from"]

# def is_navigation_request(utterance: str) -> bool:
#     return any(k in utterance.lower() for k in NAV_KEYWORDS)


# # -----------------------------------------------------
# # Social conversation prompt
# # -----------------------------------------------------
# SOCIAL_PROMPT = """
# You are Alpaca, a friendly social assistant in a building. 
# Respond naturally, politely, and helpfully in concise manner. 
# If you detect room coorindates or room numbers
# provide coordinates. Else be conversational

# User said: "{utterance}"
# """


# # -----------------------------------------------------
# # Coordinates lookup placeholder
# # -----------------------------------------------------
# ROOM_COORDS = {
#     3141: [0.0, 17.706],
#     3170: [0.0, -9.543],
#     3160: [5.0, 10.0],  
#     3171 :  [0, -9.543],
#     3150 :  [0, 8.743],
#     3140 : [0, 17.706],
#     3161 : [4.567, -4.880]
# }

# def get_coords(room_number):
#     return ROOM_COORDS.get(room_number, None)


# # -----------------------------------------------------
# # Active conversation loop
# # -----------------------------------------------------
# def guidebot_main():
#     r = sr.Recognizer()
#     mic = sr.Microphone()

#     state("GuideBot started.")

#     while True:
#         # --- Passive listening ---
#         if not passive_listen(r, mic):
#             continue
        
#         state("Wake Word Detected — entering conversation mode.")
#         last_activity = time.time()

#         while True:
#             # --- inactivity check ---
#             if time.time() - last_activity > CONVO_INACTIVITY:
#                 state("inactivity — ending conversation mode.")
#                 break

#             utterance = capture_sentence(r, mic, timeout=COMMAND_TIMEOUT)
#             if not utterance:
#                 continue

#             last_activity = time.time()

#             # --- end conversation ---
#             if any(p in utterance.lower() for p in END_PHRASES):
#                 state("Conversation ended by user.")
#                 break

#             # --- determine intent ---
#             if is_navigation_request(utterance):
#                 # NAVIGATION PATH
#                 state("Executing Navigation Command...")
#                 result = ask_llm_for_rooms(utterance)

#                 src = result.get("source_room")
#                 tgt = result.get("target_room")

#                 # fallback source
#                 if src is None:
#                     src = memory["current_position"]
#                     result["source_room"] = src

#                 # update memory with target if known
#                 if tgt:
#                     memory["current_position"] = tgt

#                 print("\n=== Navigation Result ===")
#                 print(json.dumps(result, indent=4))

#                 # show coordinates if available
#                 src_coords = get_coords(src)
#                 tgt_coords = get_coords(tgt)
#                 if src_coords:
#                     print(f"Origin coords ({src}): {src_coords}")
#                 else:
#                     print(f"Origin coords ({src}): unknown")

#                 if tgt_coords:
#                     print(f"Destination coords ({tgt}): {tgt_coords}")
#                 else:
#                     print(f"Destination coords ({tgt}): unknown")

#             else:
#                 # SOCIAL PATH
#                 state("Responding socially...")
#                 model = genai.GenerativeModel("gemini-2.5-flash")
#                 prompt = SOCIAL_PROMPT.format(utterance=utterance)
#                 try:
#                     response = model.generate_content(prompt)
#                     print(f"GuideBot: {response.text.strip()}")
#                 except Exception as e:
#                     print("[SOCIAL LLM ERROR]", e)

#             state("[ready for next instruction — still active]")



# if __name__ == "__main__":
#     try:
#         guidebot_main()
#     except KeyboardInterrupt:
#         print("Shutting down.")

## SIMPLIFIED INTERACTION CENTERED AROUND NAVIGATION

#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai


def say(text):
    """Text-to-speech"""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def state(msg):
    print(f"\n>>> {msg}\n")
    sys.stdout.flush()


# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
GENAI_API_KEY = "AIzaSyAwRx55yf-VQ1I4ycZT6dgxCe26dREuOzI"
genai.configure(api_key=GENAI_API_KEY)

WAKE_PHRASES = ["hey alpaca", "alpaca", "hello alpaca", "can you help me", "help"]
LANGUAGE = "en-US"
COMMAND_TIMEOUT = 10

# -----------------------------------------------------
# Room coordinates
# -----------------------------------------------------
ROOM_COORDS = {
    3140: [0.0, 17.706],
    3141: [0.0, 17.706],
    3150: [0.0, 8.743],
    3160: [3.2, 7.1],
    3161: [4.567, -4.880],
    3170: [0.0, -9.543],
    3171: [0.0, -9.543],
}


def get_coords(room_number):
    return ROOM_COORDS.get(room_number, None)


# -----------------------------------------------------
# LLM extraction
# -----------------------------------------------------
EXTRACTION_PROMPT = """
You are a navigation interpreter. ONLY output JSON.

Task:
- Identify the target room number.
- Convert number-words to digits.
- Ignore irrelevant text.

Output EXACTLY this format:
{{
  "target_room": <int or null>,
}}

User said: "{utterance}"
"""


def extract_room(utterance):
    prompt = EXTRACTION_PROMPT.format(utterance=utterance)
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(prompt)
        raw = response.text.replace("```json", "").replace("```", "").strip()

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None

        data = json.loads(match.group(0))
        return data.get("target_room")

    except Exception as e:
        print("[LLM ERROR]", e)
        return None


# -----------------------------------------------------
# Speech helpers
# -----------------------------------------------------
def passive_listen(r, mic):
    """Wait for wake phrase."""
    state("Passive Listening — say 'Hey Alpaca'...")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.3)
        audio = r.listen(source, phrase_time_limit=4)

    try:
        text = r.recognize_google(audio, language=LANGUAGE).lower()
        print(f"[hearing] {text}")
        return any(w in text for w in WAKE_PHRASES)
    except:
        return False


def capture_sentence(r, mic):
    state("[Listening for navigation request...]")
    with mic as source:
        audio = r.listen(source, timeout=COMMAND_TIMEOUT, phrase_time_limit=COMMAND_TIMEOUT)

    state("[processing…]")
    try:
        text = r.recognize_google(audio, language=LANGUAGE)
        state(f"Captured: {text}")
        return text.strip()
    except:
        state("Could not understand.")
        return ""


# -----------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------
def guidebot_main():
    r = sr.Recognizer()
    mic = sr.Microphone()

    state("GuideBot started.")

    while True:
        # ---- Passive listening mode ----
        if not passive_listen(r, mic):
            continue

        # Wake word detected
        say("How can I help you?")
        utterance = capture_sentence(r, mic)

        if not utterance:
            say("Sorry, I didn't catch that.")
            continue

        # Extract target room
        target_room = extract_room(utterance)
        if not target_room:
            say("I could not find a room number in your request.")
            continue

        coords = get_coords(target_room)

        if coords is None:
            say(f"Sorry, I don't know where room {target_room} is.")
            print(f"Unknown room {target_room}")
            continue

        # Confirm to user
        say(f"Okay, guiding to room {target_room}.")
        print("\n=== Navigation Command ===")
        print(f"Destination Room: {target_room}")
        print(f"Coordinates: {coords}")
        print("==========================\n")

        # Return immediately to passive listening
        state("Returning to passive listening...")


if __name__ == "__main__":
    try:
        guidebot_main()
    except KeyboardInterrupt:
        print("Shutting down.")

