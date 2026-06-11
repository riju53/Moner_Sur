import streamlit as st
import requests
import tempfile

st.set_page_config(
    page_title="🎵 Moner Sur AI Music Generator",
    page_icon="🎵"
)

st.title("🎵 Moner Sur AI Music Generator")

st.markdown("Generate music using Hugging Face MusicGen API")

# User enters API key
hf_token = st.text_input(
    "Enter Hugging Face API Token",
    type="password",
    placeholder="hf_xxxxxxxxxxxxxxxxxxxxx"
)

# Music prompt
prompt = st.text_area(
    "Music Prompt",
    value="romantic piano melody with soft strings and emotional atmosphere",
    height=120
)

# Model selection
model_name = st.selectbox(
    "Select Model",
    [
        "facebook/musicgen-small",
        "facebook/musicgen-medium"
    ]
)

if st.button("🎶 Generate Music"):

    if not hf_token:
        st.error("Please enter your Hugging Face API token.")
        st.stop()

    if not prompt.strip():
        st.error("Please enter a music prompt.")
        st.stop()

    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    with st.spinner("Generating music... This may take a few minutes."):

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt},
                timeout=300
            )

            if response.status_code == 200:

                audio_bytes = response.content

                temp_audio = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                )

                temp_audio.write(audio_bytes)
                temp_audio.close()

                st.success("✅ Music generated successfully!")

                st.audio(temp_audio.name)

                with open(temp_audio.name, "rb") as f:
                    st.download_button(
                        "⬇️ Download Music",
                        data=f,
                        file_name="moner_sur_music.wav",
                        mime="audio/wav"
                    )

            else:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)

        except requests.exceptions.Timeout:
            st.error("Request timed out. Try again.")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Hugging Face API.")

        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
