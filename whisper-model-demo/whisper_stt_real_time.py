import os
import time
import numpy as np
import pyaudio
import whisper
import torch
from deep_translator import GoogleTranslator, exceptions
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import wave

# ✅ Load Environment Variables (For Future GPT Integration)
BASE_DIR = Path(__file__).resolve().parent.parent  # Adjust based on project structure
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None  # ✅ Initialize OpenAI API if key exists

# ✅ Initialize Whisper Model (Using "small" for speed + accuracy)
model = whisper.load_model("small")

# ✅ Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz
CHUNK = 1024  # Buffer size (Adjust for latency)
SILENCE_THRESHOLD = 500  # Volume threshold for silence detection
SILENCE_TIME = 4  # Stop recording after 4 seconds of silence

# ✅ Initialize PyAudio
p = pyaudio.PyAudio()

# ✅ Store Transcriptions and Performance Metrics
transcriptions = []
detection_times = []
transcription_times = []
translation_times = []
response_times = []
total_start_time = time.time()

# ✅ Google Translator Supported Languages
SUPPORTED_LANGUAGES = GoogleTranslator().get_supported_languages()

def record_audio():
    """Records audio in real-time until silence for {SILENCE_TIME} seconds is detected."""
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("\n🎙️ Speak now... (Recording...)")
    
    frames = []
    silent_chunks = 0
    recording_start = time.time()

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

    recording_end = time.time()
    recording_duration = round(recording_end - recording_start, 2)
    
    # Stop & close the stream
    stream.stop_stream()
    stream.close()

    # Save audio to WAV file
    temp_wav = "temp_audio.wav"
    with wave.open(temp_wav, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"✅ Audio saved: {temp_wav}")
    return temp_wav, recording_duration

def transcribe(audio_file):
    """Transcribes speech using Whisper."""
    start_time = time.time()  # ✅ Start timing
    result = model.transcribe(audio_file, fp16=torch.cuda.is_available())  # Use GPU if available

    detected_language = result['language']
    transcribed_text = result['text']

    end_time = time.time()  # ✅ End timing
    execution_time = round(end_time - start_time, 2)

    print(f"\n📝 Transcribed: {transcribed_text}")
    print(f"🗣️ Detected Language: {detected_language.upper()} (⏱️ {execution_time} sec)\n")

    return detected_language, transcribed_text, execution_time

def translate(text, lang):
    """Translates text to English if it's not already English, with error handling for unsupported languages."""
    try:
        if lang == "nn" or lang == "en":
            return None, 0  # No translation needed
        start_time = time.time()  # ✅ Start timing
        translated_text = GoogleTranslator(source="auto", target="en").translate(text)
        end_time = time.time()  # ✅ End timing

        execution_time = round(end_time - start_time, 2)

        print(f"🌍 Translated to English: {translated_text} (⏱️ {execution_time} sec)")
        return translated_text, execution_time

    except exceptions.NotValidPayload:
        print(f"⚠️ Translation failed: Invalid text payload for language '{lang.upper()}'.")
        return None, 0

def get_openai_response(text):
    """Sends transcribed text to OpenAI GPT-4o-mini and returns the AI-generated response."""
    try:
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # ✅ Using "gpt-4o-mini" for fast and cost-effective responses
            store=True,
            messages=[{"role": "user", "content": text}]
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"gpt response time: {elapsed_time}")
        # print(f"GP")
        return completion.choices[0].message.content.strip(), elapsed_time

    except Exception as e:
        return f"⚠️ OpenAI API Error: {str(e)}"


# ✅ Run the Full Pipeline
while True:
    try:
        audio_file, recording_time = record_audio()
        start_time = time.time()
        detected_lang, transcribed_text, transcription_time = transcribe(audio_file)
        translated_text, translation_time = translate(transcribed_text, detected_lang)

        # ✅ Use translated text if available, otherwise use original transcription
        final_text = translated_text if translated_text else transcribed_text
        if detected_lang != "nn":
            gpt_response, response_time = get_openai_response(final_text)
            print(f"\ngpt response time: {response_time}")
            print (f" GPT-Response: {gpt_response}\n")
        end_time = time.time()
        
        total_time = end_time - start_time
        

        print(f"Total time for the operation: {total_time} sec\n")

        # ✅ Store Metrics
        # transcriptions.append((detected_lang, transcribed_text, translated_text, gpt_response))
        # detection_times.append(transcription_time)
        # transcription_times.append(transcription_time)
        # translation_times.append(translation_time)
        # response_times.append(response_time)

    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
        break