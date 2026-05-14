import streamlit as st

from downloader import download_audio
from transcriber import transcribe
from summarizer import summarize


st.set_page_config(
    page_title="YouTube Video Summarizer",
    layout="centered"
)

st.title("🎥 YouTube Video Summarizer")

youtube_url = st.text_input(
    "Enter YouTube URL"
)

if st.button("Summarize Video"):

    if not youtube_url:
        st.warning("Please enter a YouTube URL")
    else:

        with st.spinner("Downloading audio..."):
            audio_path = download_audio(youtube_url)

        with st.spinner("Transcribing video..."):
            transcript = transcribe(audio_path)

        with st.spinner("Generating summary..."):

            # limit transcript for MVP
            summary = summarize(transcript[:12000])

        st.success("Summary Generated!")

        st.subheader("Summary")

        st.write(summary)