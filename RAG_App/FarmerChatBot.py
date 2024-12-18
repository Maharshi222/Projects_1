#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import re
import logging
import openai
import os
import numpy as np

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = "################################"
openai.api_version = "################"
openai.api_key = "######################################"

# Load CSV file
csv_file_path = "FarmerAppAdd.csv"
df = pd.read_csv(csv_file_path)

# Initialize embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
questions = df["Question"].tolist()
answers = df["Answer"].tolist()
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Language detection and processing functions
def detect_language(query):
    if re.search(r'\b(kese|kya|kab|kaise|kisko|kaha|main|mujhe|me|chahna|kisan|karna)\b', query, re.I):
        if re.search(r'[a-zA-Z]', query):
            return "hi-Latn"
    if re.search(r'\b(kyare|kem|shu|tame|chhe|khedut|maari|mara|joy|dekhay|shakay)\b', query, re.I):
        if re.search(r'[a-zA-Z]', query):
            return "gu-Latn"
    detected_lang = detect(query)
    logging.info(f"Detected language: {detected_lang}")
    return detected_lang

def transliterate_to_native(input_text, language):
    if language == "Romanized Hindi":
        return transliterate(input_text, sanscript.ITRANS, sanscript.DEVANAGARI)
    elif language == "Romanized Gujarati":
        return transliterate(input_text, sanscript.ITRANS, sanscript.GUJARATI)
    return input_text

def transliterate_to_roman(input_text, language):
    if language.lower() == 'hi':
        return transliterate(input_text, sanscript.DEVANAGARI, sanscript.ITRANS)
    elif language.lower() == 'gu':
        return transliterate(input_text, sanscript.GUJARATI, sanscript.ITRANS)
    raise ValueError("Unsupported language. Please choose either 'Hindi' or 'Gujarati'.")

def query_rag(user_question):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings).squeeze()
    best_match_idx = similarities.argmax()
    return answers[best_match_idx]

from openai import AzureOpenAI
client = AzureOpenAI(
    api_key="############################",  
    api_version="################",
    azure_endpoint="#####################################"
)

def rewrite_romanized_text(input_text, language):
    prompt = f"""
    Rewrite the following text into better Romanized {language}:\n\n{input_text}\n\n
    You are an expert in rewriting text. Replace words in the input text based on the following mapping:
    - "aipa" → "app"
    - "kisana" → "kisaan"
    - "dudha" → "doodh"
    - "mem" → "mein"
    - "esaenaepha" → "SNF"
    """
    response = client.chat.completions.create(
        model="testing",
        messages=[{"role": "user", "content": prompt}]
    )
    rewritten_text = response.choices[0].message.content
    return rewritten_text

def process_query(input_text):
    detected_language = detect_language(input_text)
    st.write(f"Detected Language: {detected_language}")

    if detected_language == "hi-Latn":
        input_text = transliterate_to_native(input_text, "Romanized Hindi")
        detected_language = "hi"
        st.write(f"Transliterated to Native Script: {input_text}")
    elif detected_language == "gu-Latn":
        input_text = transliterate_to_native(input_text, "Romanized Gujarati")
        detected_language = "gu"
        st.write(f"Transliterated to Native Script: {input_text}")

    if detected_language == "Unknown":
        st.error("Unsupported language detected.")
        return "Unsupported language"

    translated_input = GoogleTranslator(source='auto', target='en').translate(input_text)
    st.write(f"Translated Input to English: {translated_input}")

    answer = query_rag(translated_input)
    st.write(f"Retrieved Answer in English: {answer}")

    if detected_language != "en":
        target_language_code = detected_language[:2]
        translated_answer = GoogleTranslator(source='en', target=target_language_code).translate(answer)
        st.write(f"Translated Answer to Detected Language: {translated_answer}")

        if target_language_code in ["hi", "gu"]:
            romanized_answer = transliterate_to_roman(translated_answer, target_language_code)
            st.write(f"Indic Roman Answer: {romanized_answer}")
            enhanced_output = rewrite_romanized_text(romanized_answer, target_language_code.upper())
            return enhanced_output
        else:
            return translated_answer
    else:
        return answer

# Streamlit UI
st.title("Multilingual Query Processor for Farmer App")
st.write("Enter your question, and the system will detect the language, process the query, and provide an answer.")

input_text = st.text_input("Enter your query:", placeholder="Type your question here...")
if st.button("Process Query"):
    if input_text.strip():
        result = process_query(input_text)
        st.success(f"Processed Output: {result}")
    else:
        st.warning("Please enter a valid query.")


# Example usage
input_text = input("Enter your query: ")
output_answer = process_query(input_text)
print("**********************************************************************************************************")
print(f"Output Answer: {output_answer.lower()}")





