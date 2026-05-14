from downloader import download_audio
from transcriber import transcribe
from summarizer import summarize


def main():

    url = input("Enter YouTube URL: ")

    print("Downloading audio...")
    audio_path = download_audio(url)

    print("Transcribing...")
    transcript = transcribe(audio_path)

    print("Generating summary...")
    summary = summarize(transcript[:12000])

    print("\\n===== SUMMARY =====\\n")
    print(summary)


if __name__ == "__main__":
    main()