import whisper
import pyaudio
import numpy as np
import torch
import time

# ✅ Load Whisper small model (Optimized for CPU)
model = whisper.load_model("small")

# ✅ Audio Parameters
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1
RATE = 16000  # Whisper requires 16kHz
CHUNK = 1024  # Process small chunks (Lower for low latency)
BUFFER_SECONDS = 5  # Start transcription after 2 seconds
BUFFER_SIZE = RATE * BUFFER_SECONDS  # Number of samples before processing

# ✅ Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# ✅ Tracking Timings & Full Transcription
full_transcription = ""
full_translation = ""
detection_times = []
transcription_times = []
translation_times = []
recording_times = []
session_start_time = time.time()  # Track full program runtime

print("🎙️ Real-Time Transcription Started... Press Ctrl+C to stop.")

def transcribe_audio():
    """ Continuously capture & transcribe audio in real-time. """
    global full_transcription, full_translation
    frames = []

    try:
        while True:
            record_start_time = time.time()  # ⏱️ Start recording time

            # ✅ Capture Audio Chunk
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(audio_chunk, dtype=np.int16))

            # ✅ Convert to NumPy & Normalize
            audio_data = np.concatenate(frames, axis=0).astype(np.float32)
            audio_data /= np.max(np.abs(audio_data))  # Normalize

            record_end_time = time.time()  # ⏱️ End recording time
            recording_times.append(record_end_time - record_start_time)

            # ✅ Transcribe Speech (Start after 2s of audio)
            if len(audio_data) > BUFFER_SIZE:
                frames = []  # Clear buffer after processing
                
                # ✅ Language Detection & Transcription
                detect_start_time = time.time()
                result = model.transcribe(audio_data, fp16=False)  # ✅ Force FP32 for CPU
                detect_end_time = time.time()

                transcribed_text = result['text']
                detected_language = result['language']

                detection_times.append(detect_end_time - detect_start_time)  # Store timing

                print(f"\n🗣️ Detected Language: {detected_language.upper()} (⏱️ {detection_times[-1]:.2f} sec)")
                print(f"📝 Transcription: {transcribed_text}")

                # ✅ Store Transcription
                full_transcription += f"{transcribed_text} "

                # ✅ Translation if needed
                translation_time = 0
                if detected_language != "en":
                    translate_start_time = time.time()
                    translated_text = model.transcribe(audio_data, task="translate")["text"]
                    translate_end_time = time.time()
                    translation_time = translate_end_time - translate_start_time  # Store timing

                    full_translation += f"{translated_text} "  # Store translation

                    print(f"🌍 Translated to English: {translated_text} (⏱️ {translation_time:.2f} sec)")

                transcription_times.append(detect_end_time - detect_start_time)  # Store transcription time
                translation_times.append(translation_time)  # Store translation time

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # ✅ Summary of Results
        session_end_time = time.time()
        total_runtime = session_end_time - session_start_time
        # avg_detection = sum(detection_times) / len(detection_times) if detection_times else 0
        avg_transcription = sum(transcription_times) / len(transcription_times) if transcription_times else 0
        avg_translation = sum(translation_times) / len(translation_times) if translation_times else 0
        avg_recording = sum(recording_times) / len(recording_times) if recording_times else 0

        print("\n🔹 **Final Transcription & Performance Summary** 🔹")
        print(f"\n📝 Full Transcription: {full_transcription.strip()}")
        if full_translation:
            print(f"🌍 Full Translation: {full_translation.strip()}")

        print("\n🔹 **Performance Metrics** 🔹")
        print(f"🎙️ Avg Recording Time: {avg_recording:.2f} sec")
        # print(f"🌍 Avg Language Detection Time: {avg_detection:.2f} sec")
        print(f"📝 Avg Detection + Transcription Time: {avg_transcription:.2f} sec")
        print(f"🌍 Avg Translation Time: {avg_translation:.2f} sec")
        print(f"🚀 Total Execution Time: {total_runtime:.2f} sec")

# ✅ Start Real-Time Transcription
transcribe_audio()