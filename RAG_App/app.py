import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator
import logging

# Constants
CSV_PATH = "FarmerAppAdd.csv"
MODEL_NAME = 'all-MiniLM-L6-v2'
AZURE_API_KEY = "##########################"
AZURE_API_VERSION = "#########################"
AZURE_ENDPOINT = "###############openai.azure.com/##################"

# Load data
df = pd.read_csv(CSV_PATH)
questions = df['Question'].tolist()
answers = df['Answer'].tolist()

# Load embedding model
model = SentenceTransformer(MODEL_NAME)
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Configure Azure OpenAI client
client = AzureOpenAI(api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_ENDPOINT)

def query_rag(user_question: str) -> str:
    """
    Retrieve the most relevant answer using semantic similarity.

    Args:
        user_question (str): The user's question.

    Returns:
        str: The most relevant answer.
    """
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings).squeeze()
    best_match_idx = similarities.argmax()
    return answers[best_match_idx]

def detect_language(query: str) -> str:
    """
    Detect the language of the given query.

    Args:
        query (str): The input text.

    Returns:
        str: Detected language code.
    """
    try:
        return detect(query)
    except Exception as e:
        logging.error(f"Error in language detection: {e}")
        return "unknown"

def translate_to_english(query: str) -> str:
    """
    Translate a query into English if it is in Hindi or Gujarati.

    Args:
        query (str): The input text.

    Returns:
        str: Translated text in English.
    """
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

def generate_answer(question: str, context: str, additional_prompt: str) -> str:
    """
    Generate an answer using the Azure OpenAI GPT model.

    Args:
        question (str): User's question.
        context (str): Relevant context.
        additional_prompt (str): Additional instructions for the model.

    Returns:
        str: The generated answer.
    """
    prompt = (
        f"Input Question from user: {question}\n\n"
        f"Relevant Context:\n{context}\n\n"
        f"Additional Prompt and Specific Instruction: {additional_prompt}\n"
        f"You are a data retrieval chatbot. Here is your chain of thought:\n"
        f"1. Read the user input and detect the language: English, Hindi, Gujarati, "
        f"Romanized Hindi, or Romanized Gujarati.\n"
        f"2. Analyze the input question and use the relevant context provided.\n"
        f"3. Generate a concise and accurate response in the same detected language."
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

# Streamlit App
st.title("Farmer App Chatbot")

st.markdown("""
This app provides answers to user questions based on the Farmer App dataset.
It retrieves the most relevant context using semantic similarity and generates human-like responses.
""")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Processing your question..."):
        translated_question = translate_to_english(user_question)
        context = query_rag(translated_question)
        language = detect_language(user_question)
        
        additional_prompt = f"Return the answer in {language}" if language in ["hi", "gu"] else "Return the answer"
        answer = generate_answer(user_question, context, additional_prompt)


    st.markdown(f"### User Question:\n{user_question}")
    st.markdown(f"### Translated Question:\n{translated_question}")
    st.markdown(f"### Language Detected:\n{language}")
    st.markdown(f"### Relevant Context:\n{context}")
    st.markdown(f"### Generated Answer:\n{answer}")





