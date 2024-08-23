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

    try:
        if file_format in ('mp3', 'm4a', 'ogg', 'flac'):
            audio_file = AudioSegment.from_file(file, format=file_format)
            audio_file.export(wav_file, format='wav')
            wav_file.seek(0)  # Reset file pointer to the beginning
        elif file_format == 'wav':
            wav_file.write(file.read())
            wav_file.seek(0)  # Reset file pointer to the beginning
        else:
            raise ValueError(f'Unsupported audio format: {file_format}')
    except Exception as e:
        st.error(f"An error occurred while processing the audio file: {e}")
        return None

    return wav_file

# Function to transcribe audio
def transcribe_audio(audio_data, language='en-US') -> str:
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# Function to perform speech-to-text transcription
def speech_to_text(file) -> str:
    wav_file = prepare_voice_file(file)
    if wav_file is None:
        return "Failed to process the audio file."
    
    with sr.AudioFile(wav_file) as source:
        audio_data = sr.Recognizer().record(source)
        text = transcribe_audio(audio_data)
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

# Function to identify speaker and generate dialogue
def generate_dialogue_text(input_text):
    prompt = f"""
    You are a sophisticated AI with advanced natural language understanding capabilities. I need you to analyze the following text and generate a dialogue-type text from it. 

    1. **Speaker Identification**: Identify the people involved in the conversation and label their dialogue appropriately.
    2. **Dialogue Generation**: Create a dialogue format from the transcribed text.

    Here is the text to analyze:

    {input_text}

    Please format your response as follows:

    - **Dialogue**: Provide a dialogue-type text with speaker labels.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit app
def main():
    st.title("Audio Transcription and Dialogue Generation")

    st.sidebar.header("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if uploaded_file:
        # Ask the user if the audio is a conversation or dialogue
        conversation_type = st.radio(
            "Is this audio a dialogue?",
            ("Single Person", "Dialogue")
        )

        with st.spinner("Transcribing the audio file..."):
            transcribed_text = speech_to_text(uploaded_file)
        st.subheader("Transcribed Text")
        st.write(transcribed_text)

        if conversation_type == "Dialogue" and st.button("Generate Dialogue"):
            with st.spinner("Generating dialogue..."):
                dialogue_text = generate_dialogue_text(transcribed_text)
            st.subheader("Generated Dialogue")
            st.write(dialogue_text)

        if st.button("Perform Sentiment Analysis") and "Could not" not in transcribed_text:
            with st.spinner("Analyzing the text..."):
                result = analyze_text_with_gemini(transcribed_text)
            st.subheader("Analysis Results")
            st.write(result)

if __name__ == "__main__":
    main()
