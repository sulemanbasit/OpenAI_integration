import whisper
import pyaudio
import wave
import numpy as np
import time
import threading
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator # online translator
import configuration as config # import the configuration.py file which has all the parameters and mapping

# ✅ Shared Variables
language_state = ""  # Holds detected language
language_lock = threading.Lock()  # Lock for thread safety
audio_file = "temp_audio.wav"  # Temporary WAV file
transcribe_restart_flag = threading.Event()  # ✅ Flag to restart transcription

# Records real-time audio and if there's a pause for {SILENCE_TIME} seconds then write it into WAV file
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
            print("🛑 Silence detected. Stopping recording.\n\n")
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

# Primary function is to detect language and change the state of language_state
def detect_language(audio_file):
    global language_state

    """Detects the spoken language from an audio file using Whisper."""
    start_time = time.time()  # ✅ Start Timer
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
    # print(f"🔍 Fixed Mel Shape: {mel.shape}")  # Should be (1, 80, 3000)

    # ✅ Detect Language
    _, probs = config.model.detect_language(mel)
    probs_dict = probs[0]  # Extract dictionary from list
    detected_language = max(probs_dict, key=probs_dict.get)

    full_language_name = config.LANGUAGE_MAP.get(detected_language, detected_language)  # Fallback if not in dict

    # ✅ Update `language_state` only when detection is successful
    with language_lock:
        if detected_language != language_state:
            language_state = detected_language  # Update to new detected language
            transcribe_restart_flag.set()  # ✅ Trigger restart of transcription
            print(f"🌍 Updated Language: {full_language_name} ({detected_language})")
        else:
            print(f"🌍 Language remains the same: {full_language_name} ({detected_language})")

    end_time = time.time()  # ✅ End Timer
    print(f"⏱️ Language Detection Time: {end_time - start_time:.2f} sec")  # ✅ Print Execution Time

    return detected_language

# ✅ Cache last model and language
last_model = None
last_language = None

# Function to Transcribe Audio using Vosk for faster time
def transcribe():
    """Transcribes audio using the appropriate Vosk model."""
    global language_state, last_model, last_language, transcribe_restart_flag

    # Wait until language is detected
    while not language_state:
        print("⏳ Waiting for language detection...")
        time.sleep(1)  # Small wait loop

    while True:  # Keep checking for language change
        start_time = time.time()  # ✅ Start Timer

        # ✅ If flag is set, restart transcription
        if transcribe_restart_flag.is_set():
            print("🔄 Language changed. Restarting transcription...\n")
            transcribe_restart_flag.clear()  # ✅ Reset flag
            continue  # Restart loop

        # ✅ Use cached model if language hasn't changed
        if language_state == last_language and last_model is not None:
            print(f"♻️ Using Cached Model: {last_language}")
            model = last_model
        else:
            # ✅ Load new model if language changed
            lang_model = config.VOSK_MODELS.get(language_state, config.VOSK_MODELS["en"])
            print(f"📝 Loading New Vosk Model: {lang_model} for transcription")
            print("Please wait, this might take some time...")

            model = Model(lang_model)  # ✅ Load new model
            last_model = model  # ✅ Cache new model
            last_language = language_state  # ✅ Update last language

        recognizer = KaldiRecognizer(model, 16000)

        transcription_result = []  # ✅ Store transcription text

        with wave.open(audio_file, "rb") as wf:
            while True:  # ✅ Read audio data from file
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):  # ✅ When full sentence detected
                    result = recognizer.Result()
                    transcription_result.append(result)

                # ✅ If flag is raised during processing, restart immediately
                if transcribe_restart_flag.is_set():
                    print("🔄 Language changed mid-transcription. Restarting...\n")
                    transcribe_restart_flag.clear()
                    break  # Stop current transcription & restart

        # ✅ Handle Final Transcription
        final_transcription = recognizer.FinalResult()
        if final_transcription.strip():  # ✅ Ensure it's not empty
            transcription_result.append(final_transcription)

        # ✅ Join all results together
        full_transcript_text = " ".join([eval(r)["text"] for r in transcription_result if r.strip()])

        print(f"📝 Final Transcription: {full_transcript_text}")  # ✅ Display Full Text

        # ✅ Translate if not English
        if language_state != "en":
            translated_text = GoogleTranslator(source=language_state, target="en").translate(full_transcript_text)
            print(f"🌍 Translated to English: {translated_text}")

        end_time = time.time()  # ✅ End Timer
        print(f"⏱️ Transcription Time: {end_time - start_time:.2f} sec")  # ✅ Print Execution Time

        # ✅ If no flag is raised, transcription is complete
        if not transcribe_restart_flag.is_set():
            break  # ✅ Exit loop once transcription is done

# ✅ Main Execution Loop
try:
    while True:
        temp_wav = record_audio()

        lang_thread = threading.Thread(target=detect_language, args=(temp_wav,))
        transcribe_thread = threading.Thread(target=transcribe)

        lang_thread.start()
        transcribe_thread.start()

        # join ensures that child tasks are completed before starting a new one
        lang_thread.join()
        transcribe_thread.join()

        time.sleep(2)
        print("\n🎤 Ready for next input...\n")
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    exit(0)