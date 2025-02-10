import os
import json
import streamlit as st
import numpy as np
import faiss
import speech_recognition as sr
from gtts import gTTS
from sentence_transformers import SentenceTransformer
from groq import Groq
import tempfile
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import io

# Initialize Speech Recognition
recognizer = sr.Recognizer()
recognizer.energy_threshold = 4000  
recognizer.dynamic_energy_threshold = False  
recognizer.pause_threshold = 1.0

# Loading Groq API Key 
GROQ_API_KEY = st.secrets["api_token"]
if not GROQ_API_KEY:
    st.error("Groq API key is missing. Please set it in Streamlit secrets.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def flatten_and_process_json(json_data, parent_key=''):
    kb_texts = []
    
    def process_value(key, value, prefix=''):
        if isinstance(value, dict):
            for k, v in value.items():
                process_value(k, v, f"{prefix}{key} - ")
        elif isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                kb_texts.append(f"{prefix}{key}: {', '.join(value)}")
            else:
                for item in value:
                    if isinstance(item, dict):
                        item_text = []
                        for k, v in item.items():
                            if isinstance(v, (str, int, float)):
                                item_text.append(f"{k}: {v}")
                        if item_text:
                            kb_texts.append(f"{prefix}{key}: {' | '.join(item_text)}")
        else:
            kb_texts.append(f"{prefix}{key}: {value}")

    for key, value in json_data.items():
        process_value(key, value)
    
    return kb_texts

def load_knowledge_base(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            knowledge_base = json.load(file)
            
        # Processing the knowledge base into searchable texts
        kb_texts = flatten_and_process_json(knowledge_base)
        
        # Generating embeddings and creating FAISS index
        if kb_texts:
            kb_embeddings = embedding_model.encode(kb_texts)
            dimension = kb_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Using Inner Product for cosine similarity
            
            # Normalising vectors for better cosine similarity
            faiss.normalize_L2(kb_embeddings)
            index.add(np.array(kb_embeddings, dtype=np.float32))
            
            return kb_texts, index
        return [], None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return [], None

# Load and process knowledge base
kb_texts, index = load_knowledge_base("knowledge_base.json")

def query_groq(prompt):
    try:
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for Indore city information."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def generate_response(query):
    if not index:
        return "Knowledge base is empty or not indexed properly."

    # Prepare query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search for similar contexts with more results
    k = 5  
    threshold = 0.3 
    
    scores, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    relevant_texts = []
    for score, idx in zip(scores[0], indices[0]):
        if score > threshold:
            relevant_texts.append(kb_texts[idx])
    
    if not relevant_texts:
        return query_groq(query)  # Fallback to Groq if no relevant context found

    context = "\n---\n".join(relevant_texts)
    
    prompt = f"""Based on the following information about Indore, please answer the user's question.
Consider all relevant details from the provided context to give a comprehensive answer.

Context:
{context}

Question: {query}

Please provide a detailed answer using ONLY the information available in the context above."""
    
    return query_groq(prompt)

def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")



def process_audio(audio_bytes):
    if audio_bytes:
        try:
            # Convert audio bytes to audio data using soundfile
            audio_data = io.BytesIO(audio_bytes)
            data, sample_rate = sf.read(audio_data)
            
            # Save as WAV file with some parameters
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                sf.write(temp_audio.name, data, sample_rate, format='WAV')
                temp_audio_path = temp_audio.name

            # Process the audio file
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
                
                # Trying different recognition settings
                try:
                    # Trying with increased timeout and without phrase time limit
                    text = recognizer.recognize_google(
                        audio,
                        language='en-IN',
                        show_all=False
                    )
                    return text
                except sr.UnknownValueError:
                    return "Could not understand audio. Please try again."
                except sr.RequestError:
                    return "Sorry, there was an error with the speech recognition service."
                finally:
                    # Cleanup
                    os.remove(temp_audio_path)
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return "Error processing audio. Please try again."
    return None

# Streamlit UI
st.title("Indore City Guide Chatbot")

st.markdown("""
Welcome to the Indore City Guide!  
Discover everything about Indore with ease. I can assist you with:  
- Must-visit tourist attractions  
- Best local food and restaurants    
- Healthcare  
- And much more!  

Ask me anything about Indore, and I'll be happy to help! 
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Adding voice input button and text input side by side
col1, col2 = st.columns([3, 1])

with col1:
    user_message = st.chat_input("Type your message here...")

with col2:
    # Adding audio recorder with a custom button
    audio_bytes = audio_recorder(
        text="🎤",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x",
        pause_threshold=3.0,
        sample_rate=16000,
        
        
    )
    
    if audio_bytes:
        # Processing audio and converting it to text
        transcribed_text = process_audio(audio_bytes)
        if transcribed_text and transcribed_text != "Sorry, I couldn't understand the audio." and transcribed_text != "Sorry, there was an error with the speech recognition service.":
            # Display user message
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            with st.chat_message("user"):
                st.markdown(transcribed_text)

            # Generate chatbot response
            chatbot_response = generate_response(transcribed_text)
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

            # Display chatbot response
            with st.chat_message("assistant"):
                st.markdown(chatbot_response)

            # Convert response to speech
            text_to_speech(chatbot_response)
        elif transcribed_text:
            st.error(transcribed_text)

if user_message:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # Generate chatbot response
    chatbot_response = generate_response(user_message)
    st.session_state.messages.append({"role": "assistant", "content": chatbot_response})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(chatbot_response)

    # Convert response to speech
    text_to_speech(chatbot_response)
