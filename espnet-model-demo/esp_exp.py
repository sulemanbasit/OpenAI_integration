import os
import time
import torch
import pyaudio
import wave
import soundfile as sf
import numpy as np
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForAudioClassification, AutoModelForPreTraining, AutoFeatureExtractor

# ✅ Force CPU usage for embedded systems
device = "cpu"
torch.set_num_threads(4)  # Limit CPU threads for faster inference
print(f"⚡ Running on: {device.upper()}")

# ✅ Load Language Identification (LID) Model (Optimized for Speed)
LID_MODEL_ID = "facebook/mms-lid-1024"
lid_processor = AutoFeatureExtractor.from_pretrained(LID_MODEL_ID)
lid_model = AutoModelForAudioClassification.from_pretrained(LID_MODEL_ID).to(device)

# ✅ Load `facebook/mms-300m` STT Model (Optimized for CPU)
STT_MODEL_ID = "facebook/mms-300m"
stt_processor = AutoProcessor.from_pretrained(STT_MODEL_ID)
stt_model = AutoModelForPreTraining.from_pretrained(STT_MODEL_ID).to(device)  # 🚀 Use PreTraining model

# ✅ Preload Tokenizer for Fast Language Switching
cached_tokenizers = {}

def get_tokenizer(lang):
    """Fetch tokenizer from cache or create new one."""
    if lang not in cached_tokenizers:
        try:
            stt_processor.tokenizer.set_target_lang(lang)
            cached_tokenizers[lang] = lang
            print(f"✅ Cached tokenizer for {lang.upper()}")
        except Exception as e:
            print(f"⚠️ Failed to set tokenizer for {lang.upper()}: {str(e)}")
    return cached_tokenizers[lang]

# ✅ Optimized Streaming Audio Recording & Transcription
def record_and_transcribe(filename, duration=10, rate=16000, chunk=1024):
    """Record audio and transcribe it in real-time (Low Latency)."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)

    frames = []
    print("🎙️ Listening & Transcribing in real-time...")

    start_time = time.time()
    transcription = ""

    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

        # ✅ Process & Transcribe Every 1 Second (Chunked Processing)
        if (i + 1) % (rate // chunk) == 0:  
            chunk_audio = b''.join(frames)
            transcription += process_audio_chunk(chunk_audio, rate)
            frames = []  # Reset for next chunk

    print("⏹️ Recording complete.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # ✅ Save Audio as WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    return filename, time.time() - start_time, transcription  # Return total recording time & transcription

# ✅ Detect Language (Using Only First 1 Second of Audio)
def detect_language(audio_file):
    """Detect language from first second of speech (Fast)."""
    speech, rate = sf.read(audio_file)
    
    # ✅ Use only the first second for detection
    speech_segment = speech[:rate * 1]

    inputs = lid_processor(speech_segment, sampling_rate=rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 🚀 Use Float32

    start_time = time.time()
    with torch.no_grad():
        logits = lid_model(**inputs).logits
        probabilities = F.softmax(logits, dim=-1)

    detected_lang_id = torch.argmax(probabilities, dim=-1).item()
    detected_lang = lid_model.config.id2label[detected_lang_id].lower()
    end_time = time.time()

    print(f"🌍 Detected Language: {detected_lang.upper()} (⏱️ {end_time - start_time:.3f} sec)")
    return detected_lang, end_time - start_time

# ✅ Streaming Speech-to-Text (Real-Time)
def process_audio_chunk(audio_chunk, rate):
    """Transcribe small audio chunks in real-time (Low Latency)."""
    speech_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
    inputs = stt_processor(speech_np, sampling_rate=rate, return_tensors="pt")
    inputs["input_values"] = inputs["input_values"].to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = stt_model(**inputs)  # ✅ Changed for `AutoModelForPreTraining`

    ids = torch.argmax(outputs.logits, dim=-1)[0]
    transcription = stt_processor.decode(ids)
    end_time = time.time()

    print(f"📝 Transcribed Chunk: {transcription} (⏱️ {end_time - start_time:.3f} sec)")
    return transcription

# ✅ Run Full Pipeline (Optimized)
audio_file, record_time, live_transcription = record_and_transcribe("speech.wav", duration=5)  # Reduce recording time

lid_time_start = time.time()
detected_lang, lid_time = detect_language(audio_file)
lid_time_end = time.time()

# ✅ Print Final Summary
print("\n🔹 **Performance Summary** 🔹")
print(f"🎙️ Recording Time: {record_time:.3f} sec")
print(f"🌍 Language Detection Time: {lid_time:.3f} sec")
print(f"🚀 Total Execution Time: {lid_time_end - lid_time_start + record_time:.3f} sec")
print(f"📝 Final Transcription: {live_transcription}")