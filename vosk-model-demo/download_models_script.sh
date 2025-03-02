#!/bin/sh

# Check if user provided an argument
if [ -z "$1" ]; then
    echo "❌ Error: Please specify a language (e.g., './download_models_script.sh french')"
    echo "Available language to choose are the following:"
    echo "english"
    echo "chinese"
    echo "russian"
    echo "french"
    echo "german"
    echo "spanish"
    echo "portuguese"
    echo "italian"
    echo "arabic"
    echo "farsi"
    echo "ukrainian"
    echo "japanese"
    echo "hindi"
    exit 1
fi

LANGUAGE=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# ✅ Use `case` instead of `declare -A`
case "$LANGUAGE" in
    english) URL="https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip" ;;
    chinese) URL="https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip" ;;
    russian) URL="https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip" ;;
    french) URL="https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip" ;;
    german) URL="https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip" ;;
    spanish) URL="https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip" ;;
    portuguese) URL="https://alphacephei.com/vosk/models/vosk-model-pt-fb-v0.1.1-20220516_2113.zip" ;;
    italian) URL="https://alphacephei.com/vosk/models/vosk-model-it-0.22.zip" ;;
    arabic) URL="https://alphacephei.com/vosk/models/vosk-model-ar-0.22-linto-1.1.0.zip" ;;
    farsi) URL="https://alphacephei.com/vosk/models/vosk-model-fa-0.42.zip" ;;
    ukrainian) URL="https://alphacephei.com/vosk/models/vosk-model-uk-v3-lgraph.zip" ;;
    japanese) URL="https://alphacephei.com/vosk/models/vosk-model-ja-0.22.zip" ;;
    hindi) URL="https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip" ;;
    *)
        echo "❌ Error: Language '$LANGUAGE' not found."
        echo "✅ Available languages: english, chinese, russian, french, german, spanish, portuguese, italian, arabic, farsi, ukrainian, japanese, hindi"
        exit 1
        ;;
esac

echo "🚀 Downloading Vosk model for: $LANGUAGE..."
curl -O "$URL"
echo "✅ Download complete: $URL"
echo "📤 Unzipping all zip files"
unzip *.zip
echo "✅ Unzipped fully"
echo "🧹 Clearing the folder now"
rm *.zip
echo "🚀 All set to go!"