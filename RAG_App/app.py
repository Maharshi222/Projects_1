# import streamlit as st
# import pandas as pd
# from openai import AzureOpenAI
# from sentence_transformers import SentenceTransformer, util
# from langdetect import detect
# import logging
# from deep_translator import GoogleTranslator



# # Load the CSV data
# CSV_PATH = "FarmerAppAdd.csv"
# df = pd.read_csv(CSV_PATH)

# # Extract questions and answers
# questions = df['Question'].tolist()
# answers = df['Answer'].tolist()

# # Load the embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Precompute embeddings for all questions
# question_embeddings = model.encode(questions, convert_to_tensor=True)

# # Azure OpenAI configuration
# client = AzureOpenAI(
#     api_key="aa611406c85e464d9d1f6d07ffe99f6c",
#     api_version="2024-08-01-preview",
#     azure_endpoint="https://promptaitest.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"
# )

# def query_rag(user_question):
#     """
#     Retrieve the most relevant answer using semantic similarity.
    
#     Parameters:
#         user_question (str): The question provided by the user.

#     Returns:
#         str: The best matching answer based on similarity.
#     """
#     # Compute the embedding for the user's question
#     user_embedding = model.encode(user_question, convert_to_tensor=True)
    
#     # Compute cosine similarities between user question and precomputed question embeddings
#     similarities = util.cos_sim(user_embedding, question_embeddings).squeeze()
    
#     # Find the index of the best match
#     best_match_idx = similarities.argmax()
    
#     # Retrieve the best matching answer
#     best_match_answer = answers[best_match_idx]
#     return best_match_answer

# def detect_language(query):
#     detected_lang = detect(query)
#     logging.info(f"Detected language: {detected_lang}")
#     return detected_lang

# from deep_translator import GoogleTranslator

# def H_G_to_English(query):
#     detected_lang = detect(query)
#     """
#     Translates a given query into English based on the detected language.

#     Parameters:
#         query (str): The text to be translated.
#         detected_lang (str): The detected language of the query ('hi' for Hindi, 'gu' for Gujarati).

#     Returns:
#         str: Translated text in English.
#     """
#     try:
#         if detected_lang == "hi":
#             hindi_conversion = GoogleTranslator(source="hi", target="en").translate(query)
#             print(f"Hindi Conversion to English: {hindi_conversion}")
#             return hindi_conversion
#         elif detected_lang == "gu":
#             gujarati_conversion = GoogleTranslator(source="gu", target="en").translate(query)
#             print(f"Gujarati Conversion to English: {gujarati_conversion}")
#             return gujarati_conversion
#         else:
#             print("Detected language not supported for translation.")
#             return query
#     except Exception as e:
#         print(f"Error occurred during translation: {e}")
#         return query



# def generate_answer(question, context, Additional_Prompt):
#     """
#     Generates an answer based on the user's question and a given context.

#     Parameters:
#         question (str): The user's input question.
#         context (str): The most relevant context for the question.

#     Returns:
#         str: The generated answer from the OpenAI model.
#     """
#     # Define the prompt
#     prompt = (
#         f"Input Question from user: {question}\n\n"
#         f"Relevant Context:\n{context}\n\n"
#         f"Additional Prompt and Specific Instruction{Additional_Prompt}"
#         f"You are a data retrieval chatbot. Here is your chain of thought:\n"
#         f"1. Read the user input and detect the language five language: English, Hindi, Gujarati, Romanized Hindi, Romanized Gujarati.\n"
#         f"2. Analyze the input question and use the relevant context provided.\n"
#         f"3. Generate a concise and accurate response to the user's question and return in same detected language of input question"
#     )

#     try:
#         # Call the Azure OpenAI API
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         # Extract and return the generated response
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"An error occurred: {e}"

# # Streamlit App
# st.title("Farmer App Chatbot")

# st.markdown("""
# This app provides answers to user questions based on the Farmer App dataset.
# It retrieves the most relevant context using semantic similarity and generates human-like responses.
# """)
# # Input box for user question
# user_question = st.text_input("Enter your question:")

# if user_question:
#     with st.spinner("Fetching the best context..."):
#         UserQuestion = H_G_to_English(user_question, )
#         best_context = query_rag(UserQuestion)

#     st.markdown(f"### Relevant Context:\n{best_context}")

#     with st.spinner("Generating the answer..."):
#         if detect_language(UserQuestion) == 'hi':
#             Hindi_Answer  = generate_answer(UserQuestion, best_context, 'Return the answer in Hindi')
#             #return Hindi_Answer
#         elif detect_language(UserQuestion) == 'gu':
#             Gujarati_Answer  = generate_answer(UserQuestion, best_context, 'Return the answer in Gujarati')
#             #return Gujarati_Answer
#         else:
#             answer = generate_answer(user_question, best_context, "Return the answer")
#             #return answer

#     st.markdown(f"### Generated Answer:\n{answer}")
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# File: farmer_app_chatbot.py

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
AZURE_API_KEY = "aa611406c85e464d9d1f6d07ffe99f6c"
AZURE_API_VERSION = "2024-08-01-preview"
AZURE_ENDPOINT = "https://promptaitest.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15-preview"

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





