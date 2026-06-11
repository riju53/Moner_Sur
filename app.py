import streamlit as st
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

st.title("🎵 MusicGen Stereo Large - AI Music Generator")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-large")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-stereo-large"
    )
    return processor, model

processor, model = load_model()

prompt = st.text_input("Enter music prompt", "cinematic orchestral music with drums")

duration = st.slider("Generation length (tokens)", 128, 1024, 256)

if st.button("Generate Music 🎶"):
    with st.spinner("Generating music... please wait"):
        inputs = processor(text=[prompt], return_tensors="pt")

        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=duration)

        audio = audio_values[0].cpu().numpy()

        # Save file
        sf.write("musicgen_output.wav", audio, 32000)

    st.success("Done!")

    st.audio("musicgen_output.wav")
    st.download_button(
        "Download WAV",
        data=open("musicgen_output.wav", "rb"),
        file_name="musicgen.wav"
    )
