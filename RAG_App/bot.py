
# File: farmer_app_chatbot_with_iterative_feedback.py

import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator
import logging

# Set page configuration
st.set_page_config(
    page_title="Farmer App Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            font-family: "Arial", sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #3c763d;
        }
        .sub-header {
            color: #3c763d;
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .question-box {
            border: 2px solid #3c763d;
            border-radius: 10px;
            padding: 1rem;
            background-color: #ffffff;
            margin-bottom: 1rem;
        }
        .output-box {
            border: 2px solid #3c763d;
            border-radius: 10px;
            padding: 1rem;
            background-color: #eaffea;
            margin-bottom: 1rem;
        }
        .spinner {
            color: #3c763d;
        }
        footer {
            text-align: center;
            font-size: 0.9rem;
            margin-top: 2rem;
            color: #555;
        }
        .feedback-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
        }
        .thumb-button {
            font-size: 1.5rem;
            padding: 0.5rem 1rem;
            background-color: #3c763d;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .thumb-button:hover {
            background-color: #2e5930;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with language selection and app info
st.sidebar.title("⚙️ Options & Info")
language_option = st.sidebar.radio(
    "Preferred Language:",
    ("English", "Hindi", "Gujarati")
)
st.sidebar.markdown("### ℹ️ About the App")
st.sidebar.info("""
This app uses AI-powered semantic similarity and a language model to answer farming-related questions.
You can:
- Ask questions about farmer App, PassBook, Milk Slips and more.
- Get answers in your preferred language (English, Hindi, or Gujarati).
- Provide feedback on the answers.
""")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.info(f"Dataset contains **{len(pd.read_csv('FarmerAppAdd.csv'))} Q&A pairs**.")

# Constants
CSV_PATH = "FarmerAppAdd.csv"
MODEL_NAME = 'all-MiniLM-L6-v2'
AZURE_API_KEY = "###########################"
AZURE_API_VERSION = "##################"
AZURE_ENDPOINT = "############################################"

# Load data
df = pd.read_csv(CSV_PATH)
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# Load embedding model
model = SentenceTransformer(MODEL_NAME)
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Configure Azure OpenAI client
client = AzureOpenAI(api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT)

def query_rag(user_question: str, disliked_indices: list) -> list:
    """
    Retrieve the top 3 most relevant unique answers not in the disliked indices.
    Args:
        user_question (str): The user's question.
        disliked_indices (list): List of indices already disliked by the user.
    Returns:
        list: A list of the top 3 unique answers with their indices and scores.
    """
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings).squeeze()

    sorted_indices = similarities.argsort(descending=True)

    results = []
    seen_answers = set()

    for idx in sorted_indices:
        if idx.item() not in disliked_indices and answers[idx.item()] not in seen_answers:
            results.append({"answer": answers[idx.item()], "index": idx.item(), "score": similarities[idx].item()})
            seen_answers.add(answers[idx.item()])
            if len(results) == 3:
                break

    while len(results) < 3:
        results.append({"answer": "No more matches available.", "index": -1, "score": 0.0})

    for i, result in enumerate(results):
        print(f"Top {i + 1} Match: {result['answer']} (Score: {result['score']})")

    return results

def translate_to_english(query: str) -> str:
    try:
        detected_lang = detect_language(query)
        if detected_lang == "hi":
            return GoogleTranslator(source="hi", target="en").translate(query)
        elif detected_lang == "gu":
            return GoogleTranslator(source="gu", target="en").translate(query)
        return query
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return query

def detect_language(query: str) -> str:
    try:
        return detect(query)
    except Exception as e:
        logging.error(f"Error in language detection: {e}")
        return "unknown"

# Modify the generate_answer function
def generate_answer(question: str, context: str, language: str) -> str:
    """
    Generate a response based on the user's question and context in the specified language.
    
    Args:
        question (str): The user's question.
        context (str): The context relevant to the question.
        language (str): The selected language (English, Hindi, Gujarati).
        
    Returns:
        str: The generated answer in the requested languages.
    """
    # Define the prompt to instruct the model in how to generate the answer
    if language == "English":
        additional_prompt = f"Generate the answer in English."
    elif language == "Hindi":
        additional_prompt = f"Generate the answer in Pure Hindi with devnagiri script and Romanized Hindi Both."
    elif language == "Gujarati":
        additional_prompt = f"Generate the answer in Pure Gujarati and Romanized Gujarati Both."
    else:
        additional_prompt = f"Generate the answer in English."

    prompt = (
        f"Input Question from user: {question}\n\n"
        f"Relevant Context:\n{context}\n\n"
        f"Additional Prompt and Specific Instruction: {additional_prompt}\n"
        f"You are a data retrieval chatbot. Here is your chain of thought:\n"
        f"1. Read the user input and detect the language: English, Hindi, Gujarati, "
        f"Romanized Hindi, or Romanized Gujarati.\n"
        f"2. Analyze the input question and use the relevant context provided.\n"
        f"3. Generate a concise and accurate response in the detected language(s)."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {e}")
        return f"An error occurred: {e}"
    


import os

# Function to log user interactions to a CSV file
def log_interaction_to_csv(question: str, response: str, feedback: str):
    """
    Logs user questions, generated responses, and feedback into a CSV file.

    Args:
        question (str): The user's question.
        response (str): The generated response.
        feedback (str): Feedback from the user ('Like' or 'Dislike').
    """
    log_file = "user_feedback_log.csv"
    # Check if the file exists; if not, create it with headers
    if not os.path.exists(log_file):
        with open(log_file, mode='w') as file:
            file.write("Question,Response,Feedback\n")

    # Append the interaction to the CSV file
    with open(log_file, mode='a') as file:
        file.write(f'"{question}","{response}","{feedback}"\n')

# User Question Handling
if "disliked_indices" not in st.session_state:
    st.session_state.disliked_indices = []
if "dislike_count" not in st.session_state:
    st.session_state.dislike_count = 0

# Page Title
st.markdown('<h1 class="main-title">🌾 Farmer App Chatbot 🌾</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-header">
    Welcome to the Farmer App Chatbot! This app uses cutting-edge AI to provide 
    relevant and helpful answers to your agricultural questions.
</div>
""", unsafe_allow_html=True)

# User Question Input
st.markdown('<div class="question-box">', unsafe_allow_html=True)
user_question = st.text_input("🔍 **Ask a question related to farming:**")
st.markdown('</div>', unsafe_allow_html=True)

if user_question:
    with st.spinner("🔄 Processing your question..."):
        translated_question = translate_to_english(user_question)
        context = query_rag(translated_question, st.session_state.disliked_indices)
        answer = generate_answer(user_question, context[0], language_option)

    # Display Outputs
    st.markdown('<div class="output-box">', unsafe_allow_html=True)
    st.markdown(f"**🗨️ Your Question:** {user_question}")
    st.markdown(f"**🌍 Translated Question:** {translated_question}")
    st.markdown(f"**📖 Relevant Context:** {context[0]}")
    st.markdown(f"**💡 Generated Answer(s):** {answer}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Feedback Buttons
    st.markdown('<div class="feedback-buttons">', unsafe_allow_html=True)
    if st.button("👍 I like this answer", key="like_button"):
        st.success("Thank you for your feedback!")
        log_interaction_to_csv(user_question, answer, "Like")  # Log feedback
        st.session_state.dislike_count = 0
        st.session_state.disliked_indices = []
    if st.button("👎 I don't like this answer", key="dislike_button"):
        st.session_state.dislike_count += 1
        st.session_state.disliked_indices.append(context[1])
        log_interaction_to_csv(user_question, answer, "Dislike")  # Log feedback
        if st.session_state.dislike_count < 3:
            st.warning("Fetching the next best match...")
        elif st.session_state.dislike_count == 3:
            st.error("It seems we are unable to provide a suitable answer. Could you rephrase your question for better understanding?")
            st.session_state.dislike_count = 0
            st.session_state.disliked_indices = []
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<footer>
    <p>🌟 Powered by OpenAI, Sentence Transformers, and Streamlit 🌟</p>
</footer>
""", unsafe_allow_html=True)
