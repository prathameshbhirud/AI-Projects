import yt_dlp

def download_audio(url: str):
    options = {
        "format": "bestaudio/best",
        "outtmpl": "data/%(id)s.%(ext)s",
        "quiet": True
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)