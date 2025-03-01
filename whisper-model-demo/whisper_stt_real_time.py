import whisper
import pyaudio
import numpy as np
import torch
import time

# Load Whisper model (medium is a good balance of speed and accuracy)
model = whisper.load_model("small")

# Audio Recording Parameters
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1              # Mono audio
RATE = 16000              # Whisper expects 16kHz
CHUNK = 1024              # Reduce buffer size to prevent overflow

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Start Recording from Microphone
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("🎙️ Speak now... Press Ctrl+C to stop.")

def record_audio():
    """ Capture audio from the microphone and return as a numpy array. """
    frames = []
    record_start_time = time.time()
    
    while time.time() - record_start_time < 10:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)  # ✅ Fix buffer overflow
            frames.append(np.frombuffer(data, dtype=np.int16))
        except OSError as e:
            print(f"⚠️ Warning: Audio buffer overflow - {e}")
            break  # Prevent loop freezing

    record_end_time = time.time()
    record_duration = record_end_time - record_start_time  # ⏱️ Time recording

    # Convert to numpy array
    audio_data = np.concatenate(frames, axis=0).astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))  # Normalize audio
    return audio_data, record_duration

try:
    while True:
        # Step 1️⃣: Capture Audio 🎙️
        print("recording audio")
        audio_input, record_time = record_audio()
        print("audio recorded\n")

        if len(audio_input) == 0:
            print("⚠️ No valid audio recorded, skipping transcription.")
            continue

        # Step 2️⃣: Transcription & Language Detection 📝
        transcribe_start_time = time.time()  # Start timer for transcription
        print("transcribing started")

        result = model.transcribe(audio_input, fp16=False)  # ✅ No more warnings
        
        transcribe_end_time = time.time()
        transcribe_time = transcribe_end_time - transcribe_start_time  # ⏱️ Time for transcription
        
        detected_language = result['language']
        transcribed_text = result['text']
        print("transcribing finished\n")

        print(f"🗣️ Detected Language: {detected_language.upper()} (⏱️ {transcribe_time:.2f} sec)")
        print(f"📝 Transcription: {transcribed_text}")

        # Step 3️⃣: Translation (If Needed) 🌍
        translate_time = 0  # Default if no translation occurs
        if detected_language != "en":
            translate_start_time = time.time()
            translated_text = model.transcribe(audio_input, task="translate")["text"]
            translate_end_time = time.time()
            translate_time = translate_end_time - translate_start_time  # ⏱️ Time for translation
            
            print(f"🌍 Translated to English: {translated_text} (⏱️ {translate_time:.2f} sec)")

        print("-" * 50)

        # 🚀 Total Execution Time
        total_time = record_time + transcribe_time + translate_time
        print("\n🔹 **Performance Summary** 🔹")
        print(f"🎙️ Recording Time: {record_time:.2f} sec")
        print(f"📝 Transcription Time: {transcribe_time:.2f} sec")
        if detected_language != "en":
            print(f"🌍 Translation Time: {translate_time:.2f} sec")
        print(f"🚀 Total Execution Time: {total_time:.2f} sec")
        print("-" * 50)

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()

