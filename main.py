import streamlit as st
import torch
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor)
import librosa
import sounddevice as sd
import wavio
import matplotlib.pyplot as plt
from pathlib import Path
import uuid
import os


def record(duration=5, fs=16000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording


def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes


def create_spectrogram(voice_sample):
    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f"Spectrogram of file {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(212)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    return fig


def transcribe(uploaded_file):
    model_load_state = st.text("Loading pretrained models...")

    processor = Wav2Vec2Processor.from_pretrained("shahukareem/wav2vec2-large-xlsr-53-dhivehi")
    model = Wav2Vec2ForCTC.from_pretrained("shahukareem/wav2vec2-large-xlsr-53-dhivehi")

    model_load_state.text("Loaded pretrained models!")
    audio_input = uploaded_file

    # transcribe
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    st.write(transcription)
    return transcription


def main():
    st.title("Dhivehi speech to text using wav2vec demo")

    input_type = st.selectbox('Input type',
                          ['','Record from microphone', 'Upload an audio file'], key='1', format_func=lambda x: 'Select an option' if x == '' else x)

    if input_type == 'Upload an audio file':
        uploaded_file = st.file_uploader("Upload an audio file")

        if uploaded_file is None:
            st.info("Please upload an audio file")
            st.stop()
        if uploaded_file:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/mp3')

            # path_myrecording = f"files/{uploaded_file.name}"
            path_myrecording = f"{uploaded_file.name}"

            with open(os.path.join(uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

            y, sr = librosa.load(path_myrecording)
            transcribe(y)

            os.remove(path_myrecording)

    elif input_type == 'Record from microphone':
        filename = str(uuid.uuid4())

        if st.button(f"Click to Record"):
            if filename == "":
                st.warning("Choose a filename.")
            else:
                record_state = st.text("Recording...")
                duration = 5  # seconds
                fs = 16000
                myrecording = record(duration, fs)
                record_state.text(f"Saving sample as {filename}.mp3")

                # path_myrecording = f"files/{filename}.mp3"
                path_myrecording = f"{filename}.mp3"

                save_record(path_myrecording, myrecording, fs)

                st.audio(read_audio(path_myrecording))

                fig = create_spectrogram(path_myrecording)
                st.pyplot(fig)
                y, sr = librosa.load(path_myrecording)
                transcribe(y)

                os.remove(path_myrecording)


if __name__ == '__main__':
    main()
