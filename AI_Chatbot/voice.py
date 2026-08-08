"""
voice.py
--------
Optional voice input/output (bonus feature).

Requires:
    pip install SpeechRecognition pyttsx3 pyaudio

These are heavy, platform-dependent dependencies (pyaudio especially
needs system audio libraries), so this module is imported lazily and
degrades gracefully: if the packages aren't installed, listen()/speak()
just tell the user voice mode isn't available instead of crashing the
whole chatbot.
"""

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


def voice_available() -> bool:
    """True if both speech-to-text and text-to-speech deps are installed."""
    return _SR_AVAILABLE and _TTS_AVAILABLE


def listen(timeout: int = 5) -> str:
    """
    Capture audio from the default microphone and transcribe it to text
    using Google's free speech recognition endpoint.
    Returns an empty string if voice input isn't available or recognition fails.
    """
    if not _SR_AVAILABLE:
        print("[voice.py] speech_recognition not installed - voice input disabled.")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening... speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=timeout)
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"[voice.py] Could not understand audio: {e}")
        return ""


def speak(text: str):
    """Speak `text` out loud using the system TTS engine, if available."""
    if not _TTS_AVAILABLE:
        print("[voice.py] pyttsx3 not installed - voice output disabled.")
        return

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    if voice_available():
        spoken = listen()
        if spoken:
            speak(f"You said: {spoken}")
    else:
        print("Voice packages not installed. Run:")
        print("  pip install SpeechRecognition pyttsx3 pyaudio")
