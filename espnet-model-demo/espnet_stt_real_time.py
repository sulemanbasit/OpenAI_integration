from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import pyaudio
import numpy as np
import queue
import torchaudio

# ✅ Load ESPnet Whisper-based Model
MODEL_NAME = "espnet/owsm_v3"  # Whisper-based ESPnet model
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# ✅ Configure Audio Stream for Low Latency
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1              # Mono audio
RATE = 16000              # 16kHz (optimized for ASR)
CHUNK = 4000              # Adjust for lower latency
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
    while True:
        while not audio_queue.empty():
            data = audio_queue.get()
            audio_array = np.frombuffer(data, dtype=np.int16)  # Convert to NumPy array
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(device)

            # ✅ Run ESPnet Whisper-based Model for STT
            input_features = processor(audio_tensor, sampling_rate=RATE, return_tensors="pt").input_features.to(device)
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            if transcription.strip():  # Ignore empty responses
                print("📝 Transcribed Text:", transcription)

except KeyboardInterrupt:
    print("\n🛑 Stopping transcription...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()