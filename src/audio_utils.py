import io
from io import BytesIO

from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# pydub needs ffmpeg on PATH to convert recorded audio to WAV. Rather than
# rely on the system PATH (unreliable on Windows), point it at the ffmpeg
# binary bundled with imageio-ffmpeg, which requires no manual setup.
try:
    import imageio_ffmpeg

    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = _ffmpeg_path
    AudioSegment.ffmpeg = _ffmpeg_path
    AudioSegment.ffprobe = _ffmpeg_path
except ImportError:
    # Falls back to whatever ffmpeg is on PATH if imageio-ffmpeg isn't installed.
    pass


def text_to_speech(text: str) -> BytesIO:
    """Converts a text report into an in-memory MP3 audio buffer (Voice Synthesis)."""
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


def speech_to_text(audio_segment: AudioSegment) -> str:
    """Transcribes a recorded audio segment (from the mic recorder) into text.

    Uses Google's free web speech API via SpeechRecognition. Returns an empty
    string if nothing could be understood, so the caller can show a friendly
    warning instead of crashing.
    """
    if audio_segment is None or len(audio_segment) == 0:
        return ""

    recognizer = sr.Recognizer()
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)

    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""