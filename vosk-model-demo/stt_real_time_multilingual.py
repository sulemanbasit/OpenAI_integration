import whisper
import pyaudio
import wave
import numpy as np
import time

# Language map from whisper probability key
LANGUAGE_MAP = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "eo": "Esperanto",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "sn": "Shona", "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "zh": "Chinese"
}

# ✅ Load Whisper Model (Choose small/tiny for speed)
model = whisper.load_model("small")

# ✅ Audio Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16kHz
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Adjust based on mic sensitivity
SILENCE_TIME = 1  # Stop recording after 1 second of silence

def record_audio():
    """Records audio until 1 second of silence is detected."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, frames_per_buffer=CHUNK)

    print("🎙️ Speak now... (Press Ctrl+C to stop)")
    
    frames = []
    silent_chunks = 0
    recording = True

    while recording:
        data = stream.read(CHUNK)
        frames.append(data)

        # Convert audio chunk to NumPy array for silence detection
        audio_np = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_np).mean()

        if volume < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0  # Reset silence counter if speech detected

        # Stop if silence lasts more than `SILENCE_TIME`
        if silent_chunks > (SILENCE_TIME * RATE / CHUNK):
            print("🛑 Silence detected. Stopping recording.")
            break

    # Stop & close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio as a WAV file
    temp_wav = "temp_audio.wav"
    wf = wave.open(temp_wav, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
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
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # ✅ Ensure Proper Shape: (1, 80, 3000)
    mel = mel.unsqueeze(0)  # Add batch dimension

    # ✅ Debug Output Shape
    print(f"🔍 Fixed Mel Shape: {mel.shape}")  # Should be (1, 80, 3000)

    # ✅ Detect Language
    _, probs = model.detect_language(mel)
    probs_dict = probs[0]  # Extract dictionary from list
    detected_language = max(probs_dict, key=probs_dict.get)  # ✅ Correct

    full_language_name = LANGUAGE_MAP.get(detected_language, detected_language)  # Fallback if not in dict
    print(f"🌍 Detected Language: {full_language_name}")

    time.sleep(2) # stop for 2 seconds

    return detected_language

# ✅ Main Execution Loop
while True:
    temp_wav = record_audio()
    detect_language(temp_wav)
    # detect_language("temp_audio.wav")
    print("\n🎤 Ready for next input...\n")