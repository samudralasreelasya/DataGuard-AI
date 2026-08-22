from io import BytesIO
from gtts import gTTS

def text_to_speech(text: str) -> BytesIO:
    """Converts text report into an in-memory MP3 audio buffer."""
    tts = gTTS(text=text, lang='en', slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer