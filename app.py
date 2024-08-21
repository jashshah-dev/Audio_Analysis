import streamlit as st
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-pro')

# Function to prepare audio file from BytesIO object
def prepare_voice_file(file) -> io.BytesIO:
    file_format = os.path.splitext(file.name)[1][1:]
    wav_file = io.BytesIO()

    if file_format in ('mp3', 'm4a', 'ogg', 'flac'):
        audio_file = AudioSegment.from_file(file, format=file_format)
        audio_file.export(wav_file, format='wav')
        wav_file.seek(0)  # Reset file pointer to the beginning
    elif file_format == 'wav':
        wav_file.write(file.read())
        wav_file.seek(0)  # Reset file pointer to the beginning
    else:
        raise ValueError(f'Unsupported audio format: {file_format}')
    
    return wav_file

# Function to transcribe audio
def transcribe_audio(audio_data, language) -> str:
    r = sr.Recognizer()
    text = r.recognize_google(audio_data, language=language)
    return text

# Function to perform speech-to-text transcription
def speech_to_text(file) -> str:
    wav_file = prepare_voice_file(file)
    with sr.AudioFile(wav_file) as source:
        audio_data = sr.Recognizer().record(source)
        text = transcribe_audio(audio_data, language='en-US')
        return text

# Function to analyze text with Google Gemini
def analyze_text_with_gemini(input_text):
    prompt = f"""
    You are a sophisticated AI with advanced natural language understanding capabilities. I need you to analyze the following text for two main tasks:

    1. **Entity Recognition**: Identify and list all entities mentioned in the text. Entities can include people, organizations, locations, products, or any specific terms relevant to the context.

    2. **Sentiment Analysis**: Analyze the sentiment expressed in the text. Provide a general sentiment overview and, if possible, identify sentiments related to specific entities or components mentioned.

    Here is the text to analyze:

    {input_text}

    Please format your response as follows:

    - **Entities Identified**: List of entities with their types.
    - **Sentiment Overview**: General sentiment expressed in the text.
    - **Entity-Specific Sentiments**: If applicable, sentiment related to each identified entity.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit app
def main():
    st.title("Audio Transcription and Sentiment Analysis")

    st.sidebar.header("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if uploaded_file:
        st.write("Transcribing the audio file...")
        transcribed_text = speech_to_text(uploaded_file)
        st.subheader("Transcribed Text")
        st.write(transcribed_text)

        if st.button("Perform Sentiment Analysis"):
            st.write("Analyzing the text...")
            result = analyze_text_with_gemini(transcribed_text)
            st.subheader("Analysis Results")
            st.write(result)

if __name__ == "__main__":
    main()
