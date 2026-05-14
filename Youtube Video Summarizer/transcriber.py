from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cpu")

def transcribe(audio_path: str):
    segments, info = model.transcribe(audio_path)

    transcript = ""

    for segment in segments:
        transcript += segment.text + " "

    return transcript