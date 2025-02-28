import os
import time
import queue
import torch
import pyaudio
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForCTC, AutoFeatureExtractor

# ✅ Load Language Identification (LID) Model (Detects 1024 languages)
LID_MODEL_ID = "facebook/mms-lid-1024"
lid_processor = AutoFeatureExtractor.from_pretrained(LID_MODEL_ID)
lid_model = AutoModelForAudioClassification.from_pretrained(LID_MODEL_ID)

# ✅ Load Multi-Language Speech-to-Text Model (1100+ languages)
STT_MODEL_ID = "facebook/mms-1b-all"
stt_processor = AutoProcessor.from_pretrained(STT_MODEL_ID)
stt_model = AutoModelForCTC.from_pretrained(STT_MODEL_ID)

# ✅ Configure Audio Input
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1              # Mono audio
RATE = 16000              # 16kHz (Required for model)
CHUNK = 2000              # Optimized for low latency
RECORD_SECONDS = 10       # Adjust as needed
OUTPUT_FILENAME = "speech.wav"

audio_queue = queue.Queue()
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    """Callback function for real-time audio processing."""
    audio_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

# ✅ Start the Microphone Stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, stream_callback=callback)

print("🎙️ Listening... Speak in any supported language.")

stream.start_stream()

try:
    frames = []
    start_time = time.time()

    # Record audio for the specified duration
    while time.time() - start_time < RECORD_SECONDS:
        while not audio_queue.empty():
            frames.append(audio_queue.get())

    print("⏹️ Recording finished. Processing...")

    # ✅ Stop audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # ✅ Convert recorded data to NumPy array
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # ✅ Save as WAV file for processing
    sf.write(OUTPUT_FILENAME, audio_np, RATE)

    # ✅ Load and process audio for Language Identification (LID)
    speech, rate = sf.read(OUTPUT_FILENAME)
    lid_inputs = lid_processor(speech, sampling_rate=RATE, return_tensors="pt")

    # ✅ Predict Language
    with torch.no_grad():
        lid_outputs = lid_model(**lid_inputs).logits

    detected_lang_id = torch.argmax(lid_outputs, dim=-1).item()
    detected_lang = lid_model.config.id2label[detected_lang_id]

    print(f"🌍 Detected Language: {detected_lang.upper()}")

    # ✅ Load the correct language adapter dynamically
    stt_processor.tokenizer.set_target_lang(detected_lang)
    stt_model.load_adapter(detected_lang)

    # ✅ Transcribe Speech in Detected Language
    stt_inputs = stt_processor(speech, sampling_rate=RATE, return_tensors="pt")

    with torch.no_grad():
        outputs = stt_model(**stt_inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = stt_processor.decode(ids)

    print(f"📝 Transcribed Text ({detected_lang.upper()}):", transcription)

except KeyboardInterrupt:
    print("\n🛑 Stopping transcription...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()