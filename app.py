import streamlit as st
import requests
import tempfile

st.title("🎵 Moner Sur AI Music Generator")

HF_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "facebook/musicgen-stereo-large"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

prompt = st.text_area(
    "Enter music prompt",
    "romantic piano melody with soft strings"
)

if st.button("Generate Music 🎶"):

    with st.spinner("Generating music..."):

        payload = {
            "inputs": prompt
        }

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=600
        )

        if response.status_code == 200:

            audio_bytes = response.content

            tmp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            tmp_file.write(audio_bytes)
            tmp_file.close()

            st.success("Music Generated!")

            st.audio(tmp_file.name)

            with open(tmp_file.name, "rb") as f:
                st.download_button(
                    "Download WAV",
                    f,
                    file_name="music.wav",
                    mime="audio/wav"
                )

        else:
            st.error(
                f"API Error: {response.status_code}"
            )
            st.write(response.text)
