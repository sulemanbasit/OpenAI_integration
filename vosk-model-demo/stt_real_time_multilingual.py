import whisper
import pyaudio
import wave
import numpy as np
import time
import configuration as config # import the configuration.py file which has all the parameters and mapping

def record_audio():
    """Records audio until 1 second of silence is detected."""
    p = pyaudio.PyAudio()
    stream = p.open(format=config.FORMAT, channels=config.CHANNELS, rate=config.RATE, 
                    input=True, frames_per_buffer=config.CHUNK)

    print("🎙️ Speak now... (Press Ctrl+C to stop)")
    
    frames = []
    silent_chunks = 0
    recording = True

    while recording:
        data = stream.read(config.CHUNK)
        frames.append(data)

        # Convert audio chunk to NumPy array for silence detection
        audio_np = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_np).mean()

        if volume < config.SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0  # Reset silence counter if speech detected

        # Stop if silence lasts more than `SILENCE_TIME`
        if silent_chunks > (config.SILENCE_TIME * config.RATE / config.CHUNK):
            print("🛑 Silence detected. Stopping recording.")
            break

    # Stop & close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio as a WAV file
    temp_wav = "temp_audio.wav"
    wf = wave.open(temp_wav, 'wb')
    wf.setnchannels(config.CHANNELS)
    wf.setsampwidth(p.get_sample_size(config.FORMAT))
    wf.setframerate(config.RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"temp_wav: {temp_wav}")
    return temp_wav

def detect_language(audio_file):
    """Detects the spoken language from an audio file using Whisper."""
    print("🔄 Detecting Language...")
    
    audio = whisper.load_audio(audio_file)
    print("loaded audio")
    
    # ✅ Trim or Pad to Ensure Exactly 30s (Fixes Incorrect Shape)
    audio = whisper.pad_or_trim(audio)

    # ✅ Convert to Mel Spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(config.model.device)

    # ✅ Ensure Proper Shape: (1, 80, 3000)
    mel = mel.unsqueeze(0)  # Add batch dimension

    # ✅ Debug Output Shape
    print(f"🔍 Fixed Mel Shape: {mel.shape}")  # Should be (1, 80, 3000)

    # ✅ Detect Language
    _, probs = config.model.detect_language(mel)
    probs_dict = probs[0]  # Extract dictionary from list
    detected_language = max(probs_dict, key=probs_dict.get)  # ✅ Correct

    full_language_name = config.LANGUAGE_MAP.get(detected_language, detected_language)  # Fallback if not in dict
    print(f"🌍 Detected Language: {full_language_name}")

    time.sleep(2) # stop for 2 seconds

    return detected_language

# ✅ Main Execution Loop
while True:
    temp_wav = record_audio()
    detect_language(temp_wav)
    # detect_language("temp_audio.wav")
    print("\n🎤 Ready for next input...\n")