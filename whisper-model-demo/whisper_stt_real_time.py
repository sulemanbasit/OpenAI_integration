import whisper
import pyaudio
import numpy as np
import torch
import time

# Load Whisper model (medium is a good balance of speed and accuracy)
model = whisper.load_model("medium")

# Audio Recording Parameters
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1              # Mono audio
RATE = 16000              # Sample rate (Whisper uses 16kHz)
CHUNK = 1024              # Buffer size

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
    start_time = time.time()
    
    # Capture for 3 seconds
    while time.time() - start_time < 3:
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    # Convert to numpy array
    audio_data = np.concatenate(frames, axis=0).astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))  # Normalize audio
    return audio_data

try:
    while True:
        # Capture Audio
        print("recording audio")
        audio_input = record_audio()
        print("audio recorded\n")

        # Transcribe and Detect Language
        print("transcribing started")
        result = model.transcribe(audio_input, fp16=torch.cuda.is_available())  # Use GPU if available
        
        detected_language = result['language']
        transcribed_text = result['text']
        print("transcribing finished\n")
        
        print(f"🗣️ Detected Language: {detected_language.upper()}")
        print(f"📝 Transcription: {transcribed_text}")
        
        # Translate to English if needed
        if detected_language != "en":
            translated_text = model.transcribe(audio_input, task="translate")["text"]
            print(f"🌍 Translated to English: {translated_text}")

        print("-" * 50)

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()