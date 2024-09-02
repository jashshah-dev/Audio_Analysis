import streamlit as st
from pydub import AudioSegment
import tempfile
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Google API for audio processing
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def transcribe_audio(audio_file_path):
    """Transcribe the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Transcribe this dialogue in the format of timecode, speaker, caption. Use speaker A, speaker B, etc. to identify speakers. If the speaker continues, just continue with the timestamp and do not show different timestamps. Output it in a pretty dialogue format.",
            audio_file
        ]
    )
    return response.text

def summarize_audio(audio_file_path):
    """Summarize the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Summarize the audio using Google's Generative API.",
            audio_file
        ]
    )
    return response.text

def analyze_entities_and_sentiment(audio_file_path):
    """Analyze entities and sentiment in the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please identify the main entities mentioned in this audio and provide a sentiment analysis with polarity scores for each entity. Format the output as a list of entities with their associated sentiment and polarity score.The polarity score must not be a range and it should be a precise value.Since this is a healthcare domain audio we need to focus more on parts where the healthcare representative talks and the way care was given and also focus on when the customer talks about the healthcare reviews to give a proper overall sentiment of the care provided",
            audio_file
        ]
    )
    return response.text

def identify_speakers(audio_file_path):
    """Identify speakers in the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please identify the speakers in the following audio.",
            audio_file
        ]
    )
    return response.text

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Streamlit app interface
st.set_page_config(page_title="Audio Analysis App", layout="wide")

st.title('🎙️ Audio Analysis App')

with st.expander("ℹ️ About this app"):
    st.write("""
        This advanced audio analysis app utilizes Google's Generative AI to provide comprehensive insights into your audio files:
        
        1. 📝 **Transcription**: Get a detailed transcript of the dialogue with speaker identification and timestamps.
        2. 📊 **Summarization**: Receive a concise summary of the main points discussed in the audio.
        3. 🧠 **Entity and Sentiment Analysis**: Identify key entities mentioned and their associated sentiment with polarity scores.
        4. 👥 **Speaker Identification**: Recognize distinct speakers in the audio content.
        
        Upload your audio file in WAV or MP3 format to unlock these powerful analysis features!
    """)

audio_file = st.file_uploader("📤 Upload Audio File", type=['wav', 'mp3'])

if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)
    st.audio(audio_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🎭 Transcribe Audio', use_container_width=True):
            with st.spinner('Transcribing...'):
                transcription = transcribe_audio(audio_path)
                st.subheader('📜 Transcription')
                st.text_area("", transcription, height=300)
        
        if st.button('📝 Summarize Audio', use_container_width=True):
            with st.spinner('Summarizing...'):
                summary = summarize_audio(audio_path)
                st.subheader('📋 Summary')
                st.info(summary)
    
    with col2:
        if st.button('🔍 Analyze Entities & Sentiment', use_container_width=True):
            with st.spinner('Analyzing...'):
                analysis = analyze_entities_and_sentiment(audio_path)
                st.subheader('🧐 Entity & Sentiment Analysis')
                st.write(analysis)
        
        if st.button('👥 Identify Speakers', use_container_width=True):
            with st.spinner('Identifying speakers...'):
                speaker_info = identify_speakers(audio_path)
                st.subheader('🎤 Speaker Identification Results')
                st.write(speaker_info)

st.markdown("---")
st.markdown("Powered by Google's Generative AI 🚀")
