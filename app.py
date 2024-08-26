import streamlit as st
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures  # For parallel processing

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
            audio_file = audio_file.set_frame_rate(16000)  # Reduce sample rate
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

# Function to transcribe audio chunk
def transcribe_audio_chunk(chunk_file):
    with sr.AudioFile(chunk_file) as source:
        audio_data = sr.Recognizer().record(source)
        return transcribe_audio(audio_data)

# Function to transcribe audio in chunks
def transcribe_chunks(audio_file, chunk_length_ms=60000) -> str:
    audio = AudioSegment.from_file(audio_file)
    total_length = len(audio)
    transcribed_text = ""

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)
    chunk_count = (total_length // chunk_length_ms) + 1
    current_chunk = 0

    # Use concurrent futures for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for start_time in range(0, total_length, chunk_length_ms):
            end_time = min(start_time + chunk_length_ms, total_length)
            chunk = audio[start_time:end_time]

            with io.BytesIO() as chunk_file:
                chunk.export(chunk_file, format='wav')
                chunk_file.seek(0)
                futures.append(executor.submit(transcribe_audio_chunk, chunk_file))

            # Update progress bar based on elapsed time
            elapsed_time = end_time / 1000  # Convert milliseconds to seconds
            progress = elapsed_time / (total_length / 1000)  # Convert total_length to seconds
            progress_bar.progress(progress)

        # Collect results from futures
        for future in concurrent.futures.as_completed(futures):
            transcribed_text += future.result() + " "

    # Complete progress bar
    progress_bar.progress(1.0)

    return transcribed_text.strip()

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

# Function to generate dialogue from text
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

# Function to identify speakers
def identify_speakers(input_text):
    prompt = f"""
    You are a sophisticated AI with advanced natural language understanding capabilities. I need you to analyze the following text and identify the number of speakers involved in the conversation.

    Here is the text to analyze:

    {input_text}

    Please provide your response in the following format:
    - **Number of Speakers**: [Insert number]
    - **Explanation**: [Brief explanation of how you determined this]
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# Function to extract number of speakers from Gemini response
def extract_num_speakers(speakers_info):
    try:
        num_speakers = int(speakers_info.split("Number of Speakers**: ")[1].split("\n")[0])
    except:
        num_speakers = 1  # Default to 1 if parsing fails
    return num_speakers

# Function to summarize text with Google Gemini
def summarize_text(input_text):
    prompt = f"""
    You are a sophisticated AI with advanced natural language understanding capabilities. I need you to summarize the following text in 80-100 words.

    Here is the text to summarize:

    {input_text}

    Please provide a concise summary.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit app
def main():
    st.set_page_config(page_title="Audio Transcription & Analysis", layout="wide")

    st.title("🔍 Audio Transcription & Analysis Tool")
    st.markdown("""
    Welcome to the **Audio Transcription & Analysis Tool**! 

    This application allows you to:
    - **Transcribe** audio files into text.
    - **Summarize** the transcribed text.
    - **Identify speakers** and generate dialogues from conversations.
    - **Perform sentiment analysis** on the transcribed text.

    Upload your audio file using the sidebar and follow the instructions to analyze it.
    """)

    st.sidebar.header("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])

    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = None

    if uploaded_file and st.session_state.transcribed_text is None:
        st.write("Transcribing the audio file might take some time. We will split the file into chunks to manage large files efficiently.")
        
        with st.spinner("Transcribing the audio file..."):
            st.session_state.transcribed_text = transcribe_chunks(uploaded_file)

    if st.session_state.transcribed_text is not None:
        st.subheader("Transcribed Text")
        st.write(st.session_state.transcribed_text)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Summarize Text"):
                with st.spinner("Summarizing the text..."):
                    summary_text = summarize_text(st.session_state.transcribed_text)
                st.subheader("Summary")
                st.write(summary_text)

        with col2:
            if st.button("Identify Speakers"):
                with st.spinner("Identifying speakers..."):
                    speakers_info = identify_speakers(st.session_state.transcribed_text)
                st.subheader("Speaker Information")
                st.write(speakers_info)

                num_speakers = extract_num_speakers(speakers_info)
                if num_speakers > 1:
                    st.write("This appears to be a dialogue.")
                    if st.button("Generate Dialogue"):
                        with st.spinner("Generating dialogue..."):
                            dialogue_text = generate_dialogue_text(st.session_state.transcribed_text)
                        st.subheader("Generated Dialogue")
                        st.write(dialogue_text)
                else:
                    st.write("This appears to be a monologue.")

        with col3:
            if st.button("Perform Sentiment Analysis"):
                with st.spinner("Analyzing the text..."):
                    result = analyze_text_with_gemini(st.session_state.transcribed_text)
                st.subheader("Analysis Results")
                st.write(result)

if __name__ == "__main__":
    main()
