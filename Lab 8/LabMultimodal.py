import base64
from pathlib import Path
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def prepare_image(upload):
    formats = {".png": "image/png", ".jpg": "image/jpeg",
               ".jpeg": "image/jpeg"}
    mime = formats[Path(upload.name).suffix.lower()]
    encoded = base64.b64encode(upload.getvalue()).decode("utf-8")
    return {"type": "input_image",
            "image_url": f"data:{mime};base64,{encoded}"}


def transcribe_audio(upload):
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=(upload.name, upload.getvalue()))
    return transcript.text


def analyze_content(content, prompt):
    response = client.responses.create(
        model="gpt-4.1-mini", instructions=prompt,
        input=[{"role": "user", "content": content}],
        max_output_tokens=600)
    return response.output_text


# System prompt: request a summary + 3 key points, answer grounded in the
# uploaded material, and acknowledge when something can't be answered
SYSTEM_PROMPT = """You are a "See and Hear" assistant. You are given either an
image or a transcript of an audio clip as source material.

Always respond with:
1. A short summary of the content (2-3 sentences)
2. Three key points from the content, as a bulleted list, with the most
   important term in each point written in **bold**

If the user asks a question, answer it using only the uploaded material.
If the material does not contain enough information to answer, reply with
exactly this sentence: "I cannot answer this based on the provided material."
Do not use outside knowledge or guess."""

st.title("See and Hear Assistant")
st.write("Upload an image or audio clip, get a summary, and ask a question about it.")

media_type = st.radio("What are you uploading?", ["Image", "Audio"])

if media_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG) - max 5 MB",
        type=["png", "jpg", "jpeg"])
else:
    uploaded_file = st.file_uploader(
        "Upload an audio clip (MP3, WAV, M4A) - max 60 seconds",
        type=["mp3", "wav", "m4a"])

question = st.text_input(
    "Optional: ask a question about the upload",
    placeholder="Leave blank for a general summary")

MAX_IMAGE_MB = 5
MAX_AUDIO_SECONDS = 60  # not checked directly, file size used as a proxy
MAX_AUDIO_MB = 10

analyze_clicked = st.button("Analyze", disabled=not uploaded_file)

if uploaded_file and media_type == "Image":
    st.image(uploaded_file)
elif uploaded_file and media_type == "Audio":
    st.audio(uploaded_file)

if analyze_clicked:
    # Size check before calling the API
    size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    limit = MAX_IMAGE_MB if media_type == "Image" else MAX_AUDIO_MB

    if size_mb > limit:
        st.error(f"File is {size_mb:.1f} MB. Please upload a file under {limit} MB.")
    else:
        with st.spinner("Analyzing..."):
            try:
                if media_type == "Image":
                    image_block = prepare_image(uploaded_file)
                    text_block = {
                        "type": "input_text",
                        "text": question if question else "Summarize this image."}
                    content = [text_block, image_block]
                    result = analyze_content(content, SYSTEM_PROMPT)
                else:
                    transcript = transcribe_audio(uploaded_file)
                    with st.expander("Transcript"):
                        st.write(transcript)
                    text_block = {
                        "type": "input_text",
                        "text": f"Transcript: {transcript}\n\nQuestion: "
                                f"{question if question else 'Summarize this audio.'}"}
                    content = [text_block]
                    result = analyze_content(content, SYSTEM_PROMPT)

                st.write(result)
            except Exception as e:
                st.error(f"Something went wrong: {e}")