# configuration file which holds all parameters and mappings

import whisper
import pyaudio

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

# ✅ Vosk Model Paths (Downloaded & Extracted)
ABSOLUTE_PATH = "/Users/dianbasit/Library/CloudStorage/Dropbox/My Mac (Dians-MacBook-Air.local)/Documents/Upwork_Project/OpenAI_integration/vosk-model-demo"
VOSK_MODELS = {
    "en": f"{ABSOLUTE_PATH}/vosk-model-en-us-0.22",
    "fr": f"{ABSOLUTE_PATH}/vosk-model-fr-0.22",
    "es": f"{ABSOLUTE_PATH}/vosk-model-es-0.42",
    "de": f"{ABSOLUTE_PATH}/vosk-model-de-0.21",
    "it": f"{ABSOLUTE_PATH}/vosk-model-it-0.22",
    "ru": f"{ABSOLUTE_PATH}/vosk-model-ru-0.42",
    "pt": f"{ABSOLUTE_PATH}/vosk-model-pt-fb-v0.1.1-20220516_2113",
    "zh": f"{ABSOLUTE_PATH}/vosk-model-cn-0.22",
    "ar": f"{ABSOLUTE_PATH}/vosk-model-ar-0.22-linto-1.1.0",
    "fa": f"{ABSOLUTE_PATH}/vosk-model-fa-0.42",
    "hi": f"{ABSOLUTE_PATH}/vosk-model-hi-0.22",
    "ja": f"{ABSOLUTE_PATH}/vosk-model-ja-0.22",
    "uk": f"{ABSOLUTE_PATH}/vosk-model-uk-v3-lgraph",
}

# ✅ Audio Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16kHz
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Adjust based on mic sensitivity
SILENCE_TIME = 2  # Stop recording after 1 second of silence
