import os
import time
import json
import queue
import pyaudio
from vosk import Model, KaldiRecognizer

# ✅ Load Optimized Vosk Model (Ensure it's downloaded)
# MODEL_PATH = "vosk-model-small-en-us-0.15" # 50MB
MODEL_PATH = "vosk-model-en-us-0.22" # 1.9GB

if not os.path.exists(MODEL_PATH):
    print("❌ Error: Vosk model not found! Please download it.")
    exit(1)

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# ✅ Configure Audio Stream for Low Latency
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1              # Mono audio
RATE = 16000              # 16kHz (optimized for Vosk)
CHUNK = 2000              # Smaller chunk size (reduces latency)
audio_queue = queue.Queue()

# ✅ Initialize PyAudio
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    """Callback function for real-time audio processing."""
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

# ✅ Start the Microphone Stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, stream_callback=callback)

print("🎙️ Listening... Speak into the microphone.")

stream.start_stream()

try:
    start_time = None  # Initialize timing variable
    while True:
        while not audio_queue.empty():
            data = audio_queue.get()

            # ✅ Start timing when speech processing begins
            if start_time is None:
                start_time = time.time()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                transcription = result["text"]
                
                end_time = time.time()  # ✅ End timing after speech processing
                elapsed_time = end_time - start_time
                
                print(f"⏱️ Transcription Time: {elapsed_time:.2f} sec")
                print("📝 Transcribed Text:", transcription)

                start_time = None  # Reset for the next speech input

except KeyboardInterrupt:
    print("\n🛑 Stopping transcription...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()