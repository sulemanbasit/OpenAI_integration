import os
import wave
import time
import json
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator

# ✅ Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Adjust sensitivity
SILENCE_TIME = 2  # Stop recording after silence for 2 sec

# ✅ Vosk Model Path (English by Default)
ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{ABSOLUTE_PATH}/vosk-model-en-us-0.22" # 1.9GB EN
# MODEL_PATH = f"{ABSOLUTE_PATH}/vosk-model-fr-0.22" # 1.9GB FR
# MODEL_PATH = f"{ABSOLUTE_PATH}/vosk-model-ru-0.42" # 1.9GB RU

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Vosk model not found at {MODEL_PATH}")
    exit(1)

# ✅ Load Vosk Model
vosk_model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, RATE)

def record_audio():
    """Records audio in real-time until silence for {SILENCE_TIME} seconds is detected."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎙️ Speak now... (Recording...)")
    
    frames = []
    silent_chunks = 0

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        # Convert audio chunk to NumPy array for silence detection
        audio_np = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_np).mean()

        if volume < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0  # Reset silence counter if speech is detected

        # Stop if silence lasts for {SILENCE_TIME} seconds
        if silent_chunks > (SILENCE_TIME * RATE / CHUNK):
            print("🛑 Silence detected. Stopping recording.")
            break

    # Stop & close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to WAV file
    temp_wav = "temp_audio.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"✅ Audio saved: {temp_wav}")
    return temp_wav

def transcribe(audio_file):
    """Transcribes speech using Vosk and translates if non-English."""
    start_time = time.time()  # ✅ Start timing

    with wave.open(audio_file, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())

    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        transcription = result["text"]

        if transcription.strip():
            # ✅ Auto-translate if non-English
            
            translated_text = GoogleTranslator(source="auto", target="en").translate(transcription)

            end_time = time.time()  # ✅ End timing
            execution_time = round(end_time - start_time, 2)

            print(f"\n📝 Transcribed: {transcription}")
            print(f"🌍 Translated to English: {translated_text}")
            print(f"⏱️ Transcription Time: {execution_time} sec\n")
        else:
            print("⚠️ No speech detected.\n")
    else:
        print("⚠️ Could not process speech.\n")

# ✅ Run the Full Pipeline
while True:
    try:
        audio_file = record_audio()
        transcribe(audio_file)

    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
        break