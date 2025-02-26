# Using stt_env
# Python 3.9 recommended for vosk

import wave
import json
from vosk import Model, KaldiRecognizer
import os
import subprocess
from pathlib import Path
import time

# STT FILES FUNCTION

def is_wav_file(file_path):
    """
    Check if the file is in WAV format
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            return True
    except wave.Error:
        return False

def convert_to_wav_ffmpeg(input_file, output_file=None):
    """
    Convert audio file to WAV format using FFmpeg
    Returns the path to the converted file
    """
    if output_file is None:
        # Create output filename in the same directory as input file
        output_file = str(Path(input_file).with_suffix('.wav'))
    
    try:
        command = [
            'ffmpeg',
            '-i', input_file,
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '16000',
            output_file,
            '-y'  # Overwrite output file if it exists
        ]
        # Using capture_output=True to capture both stdout and stderr
        subprocess.run(command, check=True, capture_output=True)
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e.stderr.decode()}")
        return None
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg on your system.")
        return None

def transcribe_audio(audio_file_path, model_path="vosk-model-small-en-us-0.15"):
    """
    Main function to handle audio transcription
    Checks format, converts if necessary, and performs transcription
    """
    if not os.path.exists(model_path):
        print(f"Please download the model from https://alphacephei.com/vosk/models and unpack as {model_path}")
        return None

    # Check if file exists
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return None

    wav_file_path = audio_file_path
    converted = False

    # Check if the file is WAV format, if not convert it
    if not is_wav_file(audio_file_path):
        print(f"Converting {audio_file_path} to WAV format...")
        wav_file_path = convert_to_wav_ffmpeg(audio_file_path)
        if wav_file_path is None:
            print("Conversion failed.")
            return None
        converted = True

    try:
        # Load the model
        model = Model(model_path)
        
        # Open the audio file
        wf = wave.open(wav_file_path, "rb")
        
        # Check if the audio file has the right format
        if wf.getnchannels() != 1:
            print("Converting to mono channel...")
            temp_path = str(Path(wav_file_path).with_name('temp_mono.wav'))
            convert_to_wav_ffmpeg(wav_file_path, temp_path)
            wf.close()
            wf = wave.open(temp_path, "rb")
        
        # Create recognizer
        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(True)
        
        # Process audio file
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                part_result = json.loads(recognizer.Result())
                results.append(part_result.get("text", ""))
        
        # Get final bits of audio and flush the pipeline
        part_result = json.loads(recognizer.FinalResult())
        results.append(part_result.get("text", ""))
        
        # Clean up
        wf.close()
        
        # Remove temporary files if created
        if converted:
            try:
                if os.path.exists('temp_mono.wav'):
                    os.remove('temp_mono.wav')
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {e}")
        
        # Combine results
        transcript = " ".join(results)
        return transcript.strip()
    
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None
    
# END OF STT FUNCTIONS

# Main Program

audio_file = "Test2.m4a"  # Replace with your audio file path

# Make sure you have downloaded the Vosk model and specified the correct path
model_path = "vosk-model-small-en-us-0.15"  # Replace with your model path

print("Starting transcription...")
start_time = time.time()  # Record the start time
transcript = transcribe_audio(audio_file, model_path)

end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate elapsed time

if transcript:
    print("Transcription completed:")
    print(transcript)
else:
    print("Transcription failed.")

print(f"Total time: {execution_time}")