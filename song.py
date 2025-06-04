import os
import streamlit as st
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import cohere
from bark import generate_audio
from pydub import AudioSegment

# === CONFIG ===
COHERE_API_KEY = "WzqmtxpUY4ImR6cpPnwUEerPoRJYBVxGELe9YNSM"
#COHERE_API_KEY = os.getenv("WzqmtxpUY4ImR6cpPnwUEerPoRJYBVxGELe9YNSM")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is not set!")
# === SETUP ===
co = cohere.Client(COHERE_API_KEY)
musicgen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# === FUNCTIONS ===

def generate_lyrics(prompt):
    response = co.generate(prompt=prompt, max_tokens=500)
    return response.generations[0].text.strip()

def generate_music(prompt_text):
    inputs = musicgen_processor(text=[prompt_text], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio_values = musicgen_model.generate(**inputs)
    audio_np = audio_values[0].cpu().numpy().squeeze()
    sf.write("instrumental.wav", audio_np, samplerate=44100)
    return "instrumental.wav"

def synthesize_singing(lyrics):
    audio_array = generate_audio(lyrics)
    sf.write("vocals.wav", audio_array, samplerate=24000)
    return "vocals.wav"

def mix_tracks(inst_path, vocals_path, output_file="final_song.wav"):
    music = AudioSegment.from_wav(inst_path)
    vocals = AudioSegment.from_wav(vocals_path)
    mixed = music.overlay(vocals)
    mixed.export(output_file, format="wav")
    return output_file

# === STREAMLIT UI ===
st.title("🎵 AI Generative Music App")

title = st.text_input("Song Title", "Dreamscape Journey")
genre = st.selectbox("Genre", ["Pop", "Hip-Hop", "EDM", "Jazz", "Classical", "Lo-Fi"])
mood = st.selectbox("Mood", ["Uplifting", "Sad", "Energetic", "Chill", "Romantic"])
bpm = st.slider("Tempo (BPM)", 60, 180, 100)

if st.button("🎶 Generate Music"):
    with st.spinner("Generating lyrics..."):
        prompt = f"Write a {genre} song titled '{title}' with a {mood} mood and {bpm} BPM. Include a chorus and meaningful verses."
        lyrics = generate_lyrics(prompt)
        st.subheader("🎤 Lyrics")
        st.write(lyrics)

    with st.spinner("Creating instrumental..."):
        inst_prompt = f"A {genre} instrumental with {mood} mood at {bpm} BPM"
        instrumental_path = generate_music(inst_prompt)

    with st.spinner("Synthesizing vocals..."):
        vocal_path = synthesize_singing(lyrics)

    with st.spinner("Mixing final song..."):
        final_path = mix_tracks(instrumental_path, vocal_path)

    st.success("✅ Your AI-generated song is ready!")
    st.audio(final_path)