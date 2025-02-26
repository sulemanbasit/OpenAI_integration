# Using stt_env
# Python 3.9 recommended for vosk

import wave
import json
from vosk import Model, KaldiRecognizer
import os
import subprocess
from pathlib import Path
import time

# Main Program
# Load Vosk model
model_path = "vosk-model-small-en-us-0.15"
model = Model(model_path)

start_time = time.time()  # Record the start time

# Open WAV file (Ensure it's 16kHz)
audio_file = "Test2.wav"
wf = wave.open(audio_file, "rb")

rec = KaldiRecognizer(model, 16000)

print("Starting transcription...")
while True:
    data = wf.readframes(1000)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)

transcript = json.loads(rec.FinalResult())["text"]
print("Transcription:", transcript)

end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate elapsed time

print(f"Total time: {execution_time}")