import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import json
import sqlite3
import logging
#from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and configurations from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")

api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
ai_model = os.getenv("AI_MODEL")

# Set up Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)


#Set up embed model
azure_embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT")
azure_embed_model = os.getenv("EMBEDING_MODEL")
azure_embed_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBED")



clientembed = AzureOpenAI(
    api_key=api_key,
    api_version=azure_embed_version,
    azure_endpoint=azure_embed_endpoint
)

# Initialize FAISS indices for "farmer" and "milk_man" with ID mapping
embedding_dimension = 1536
faiss_index_farmer = faiss.IndexIDMap2(faiss.IndexFlatL2(embedding_dimension))
faiss_index_milk_man = faiss.IndexIDMap2(faiss.IndexFlatL2(embedding_dimension))

# Metadata stores for "farmer" and "milk_man"
def load_metadata(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

metadata_store_farmer = load_metadata("metadata_store_farmer.json")
metadata_store_milk_man = load_metadata("metadata_store_milk_man.json")

def save_metadata(metadata, file_path):
    with open(file_path, "w") as f:
        json.dump(metadata, f)

def save_metadata_to_db(metadata, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY,
        file_name TEXT,
        chunk_id INTEGER,
        text TEXT
    )
    """)
    cursor.execute("DELETE FROM metadata")  # Clear existing data
    for idx, meta in enumerate(metadata):
        cursor.execute("""
        INSERT INTO metadata (id, file_name, chunk_id, text)
        VALUES (?, ?, ?, ?)
        """, (idx, meta["file_name"], meta["chunk_id"], meta["text"]))
    conn.commit()
    conn.close()

# Initialize Flask app
app = Flask(__name__)

def extract_text(file):
    file_ext = file.filename.split('.')[-1].lower()
    text = ""
    if file_ext == "pdf":
        pdf_reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in pdf_reader.pages])
    elif file_ext == "docx":
        doc = Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
    elif file_ext == "txt":
        text = file.read().decode('utf-8', errors='ignore')  # Ignore errors in text
    elif file_ext == "csv":
        df = pd.read_csv(file, encoding='latin1')
        text = "\n".join([f"Q: {row['Question']} A: {row['Answer']}" for index, row in df.iterrows()])
    else:
        raise ValueError("Unsupported file type")
    return text
import numpy as np

def preprocess_and_vectorize(text, model=azure_embed_model):
    """
    Preprocesses the text into structured question-answer chunks and computes embeddings for each chunk.
    Handles both structured Q&A format and single text input.
    
    Args:
        text (str): Input text (either Q&A pairs or single query).
        model (str): The name of the embedding model to use.
    
    Returns:
        list: A list of tuples, each containing the chunk (as a combined string) and its embedding as a numpy array.
    """
    structured_chunks = []
    embeddings = []
    
    # Check if the input contains Q&A markers
    if "Q:" in text and "A:" in text:
        for chunk in text.split("\n"):
            if chunk.startswith("Q:"):
                try:
                    question = chunk[3:chunk.index("A:")].strip()
                    answer = chunk[chunk.index("A:") + 3:].strip()
                    structured_chunks.append({"Question": question, "Answer": answer})
                except ValueError as e:
                    print(f"Skipping invalid chunk: {chunk} - Error: {e}")
    else:
        # Treat the input as a single text query
        structured_chunks.append({"Question": text, "Answer": ""})

    # Prepare embeddings
    for chunk in structured_chunks:
        combined_text = f"Q: {chunk['Question']} A: {chunk['Answer']}"
        try:
            # Call the embedding model
            response = clientembed.embeddings.create(input=[combined_text], model=model)
            embedding = response.data[0].embedding
            embeddings.append((combined_text, np.array(embedding, dtype='float32')))
        except Exception as e:
            print(f"Error generating embedding for: {combined_text} - Error: {e}")

    return embeddings

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
def generate_answer(question: str, context: str,asked_question: str) -> str:
    """
    Generate a response based on the user's question and context in the specified language.
    
    Args:
        question (str): The user's question.
        context (str): The context relevant to the question.
        language (str): The selected language (English, Hindi, Gujarati).
        
    Returns:
        str: The generated answer in the requested languages.
    """
    # # Define the prompt to instruct the model in how to generate the answer
    # if language == "English":
    #     additional_prompt = f"Generate the answer in English."
    # elif language == "Hindi":
    #     additional_prompt = f"Generate the answer in Pure Hindi with devnagiri script and Romanized Hindi Both."
    # elif language == "Gujarati":
    #     additional_prompt = f"Generate the answer in Pure Gujarati and Romanized Gujarati Both."
    # else:
    #     additional_prompt = f"Generate the answer in English."

    prompt = (

        f"Input Question from user: {question}\n\n"
        f"Relevant Context:\n{context}\n\n"
        f'f"Detect the language of:\n{asked_question}\n\n"'
        f"Generate a concise and accurate response in the detected language of {asked_question}"
        f"You are a data retrieval chatbot. Here is your chain of thought:\n"
        f"1. Read the user input and detect the language: English, Hindi, Gujarati,Romanized Hindi, or Romanized Gujarati. "
        f"2. Analyze the input question and use the relevant context provided.\n"
        f"3. Generate a concise and accurate response in the detected of {asked_question}."
    )
    
    try:
        response = client.chat.completions.create(
            model= ai_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {e}")
        return f"An error occurred: {e}"

@app.route('/farmer/upload', methods=['POST'])
def upload_file_farmer():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        file_name = file.filename

        # Remove existing entries from FAISS and metadata store
        existing_chunks = [
            i for i, meta in enumerate(metadata_store_farmer)
            if meta["file_name"] == file_name
        ]
        if existing_chunks:
            print(f"File '{file_name}' already exists. Updating...")
            faiss_ids_to_remove = np.array(existing_chunks, dtype='int64')
            faiss_index_farmer.remove_ids(faiss_ids_to_remove)
            for idx in sorted(existing_chunks, reverse=True):
                del metadata_store_farmer[idx]

        # Extract text and vectorize
        text = extract_text(file)
        vectors = preprocess_and_vectorize(text)

        # Add new entries
        for i, (chunk, vector) in enumerate(vectors):
            metadata_store_farmer.append({
                "file_name": file_name,
                "chunk_id": i,
                "text": chunk
            })
            faiss_index_farmer.add_with_ids(
                np.array([vector]),
                np.array([len(metadata_store_farmer) - 1])
            )
        dir_name = "Farmer App"
        try:
            os.mkdir(dir_name)
            print(f"Directory '{dir_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{dir_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{dir_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            dir_name = None  # Prevent further actions if directory creation fails



        # Save updated metadata to file and database
        json_file_path = os.path.join(dir_name, "metadata_store_farmer.json")
        db_file_path = os.path.join(dir_name, "metadata_store_farmer.db")
        save_metadata(metadata_store_farmer, json_file_path)
        save_metadata_to_db(metadata_store_farmer, db_file_path)



        return jsonify({
            "message": "File uploaded successfully (updated if it existed)",
            "file_name": file_name,
            "chunks_stored": len(vectors)
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

def upload_file_milk_man():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        file_name = file.filename

        # Remove existing entries from FAISS and metadata store
        existing_chunks = [
            i for i, meta in enumerate(metadata_store_milk_man)
            if meta["file_name"] == file_name
        ]
        if existing_chunks:
            print(f"File '{file_name}' already exists. Updating...")
            faiss_ids_to_remove = np.array(existing_chunks, dtype='int64')
            faiss_index_milk_man.remove_ids(faiss_ids_to_remove)
            for idx in sorted(existing_chunks, reverse=True):
                del metadata_store_milk_man[idx]

        # Extract text and vectorize
        text = extract_text(file)
        vectors = preprocess_and_vectorize(text)

        # Add new entries
        for i, (chunk, vector) in enumerate(vectors):
            metadata_store_farmer.append({
                "file_name": file_name,
                "chunk_id": i,
                "text": chunk
            })
            faiss_index_milk_man.add_with_ids(
                np.array([vector]),
                np.array([len(metadata_store_milk_man) - 1])
            )
        dir_name = "milk_man App"
        try:
            os.mkdir(dir_name)
            print(f"Directory '{dir_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{dir_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{dir_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            dir_name = None  # Prevent further actions if directory creation fails



        # Save updated metadata to file and database
        json_file_path = os.path.join(dir_name, "metadata_store_milk_man.json")
        db_file_path = os.path.join(dir_name, "metadata_store_milk_man.db")
        save_metadata(metadata_store_milk_man, json_file_path)
        save_metadata_to_db(metadata_store_milk_man, db_file_path)



        return jsonify({
            "message": "File uploaded successfully (updated if it existed)",
            "file_name": file_name,
            "chunks_stored": len(vectors)
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

def rephrase(user_question):
    prompt = (

        f"You are a helpful assistant and translator.\n\n"
        f"Original Question: {user_question}\n\n"
        f"Detect language of {user_question} from English, Hindi, Gujarati, Romanized Hindi, Romanized Gujarati\n\n"
        f"Rephrased Question in English in simple and clear and formal manner."
            
         )
    
    try:
        response = client.chat.completions.create(
            model= ai_model,
            messages=[{"role": "user", "content": prompt}]
        )
        rephrase_response = response.choices[0].message.content.strip()
        print(rephrase_response)
        return rephrase_response
    except Exception as e:
        logging.error(f"Error in OpenAI API call: {e}")
        return f"An error occurred: {e}"

@app.route('/farmer/chat', methods=['POST'])
def chat_farmer():
    data = request.json
    user_message = data.get('message', '')
    rephrase_que = rephrase(user_message)
    print(f"Rephrased query: {rephrase_que}")

    if not rephrase_que:
        return jsonify({"error": "No message provided"}), 400

    # Generate embeddings for the query
    query_embeddings = preprocess_and_vectorize(rephrase_que)
    if not query_embeddings:
        return jsonify({"error": "Failed to generate embeddings for the query."}), 400

    query_embedding = query_embeddings[0][1]
    print(f"Query embedding: {query_embedding}")

    # Check FAISS index size
    print(f"FAISS Index size: {faiss_index_farmer.ntotal}")

    # Perform FAISS search
    distances, indices = faiss_index_farmer.search(np.array([query_embedding]), k=1)
    print(f"Indices: {indices}")
    print(f"Distances: {distances}")

    # Safeguard: Ensure indices are valid
    valid_docs = [
        metadata_store_farmer[idx]['text']
        for idx in indices[0]
        if 0 <= idx < len(metadata_store_farmer) and idx != -1
    ]
    print(f"Valid documents: {valid_docs}")

    if not valid_docs:
        return jsonify({"error": "No relevant context found for the query."}), 400

    # Use the context to generate an answer
    context = " ".join(valid_docs)
    ai_response = generate_answer(rephrase_que, context,user_message )

    # # Translate response back if necessary
    # if detected_lang != "en":
    #     ai_response = GoogleTranslator(source="en", target=detected_lang).translate(ai_response)

    return jsonify({"response": ai_response})

# @app.route('/farmer/chat', methods=['POST'])
# def chat_farmer():
#     data = request.json
#     user_message = data.get('message', '')
#     if not user_message:
#         return jsonify({"error": "No message provided"}), 400

#     # Step 1: Use the LLM to rephrase the question in English
#     try:
#         rephrase_prompt = (
#             f"You are a helpful assistant and trnaslator.\n\n"
#             f"Original Question: {user_message}\n\n"
#             f"Detect language of {user_message} from English, Hindi, Gujarati, Romanized Hindi, Romanized Gujarati\n\n"
#             f"Rephrased Question in English in simple and clear and formal manner."
            
#         )
#         rephrase_response = client.chat.completions.create(
#             model=ai_model,
#             messages=[{"role": "user", "content": rephrase_prompt}]
#         )
#         rephrased_question = rephrase_response.choices[0].message.content.strip()
#         print(f"Rephrased question: {rephrased_question}")
#     except Exception as e:
#         return jsonify({"error": f"Error during rephrasing: {str(e)}"}), 500

#     # Step 2: Generate embeddings for the rephrased query
#     query_embeddings = preprocess_and_vectorize(rephrased_question)
#     if not query_embeddings:
#         return jsonify({"error": "Failed to generate embeddings for the query."}), 400

#     query_embedding = query_embeddings[0][1]

#     # Step 3: Perform FAISS search
#     distances, indices = faiss_index_farmer.search(np.array([query_embedding]), k=2)

#     # Step 4: Safeguard: Ensure indices are valid
#     valid_docs = [
#         metadata_store_farmer[idx]['text']
#         for idx in indices[0]
#         if 0 <= idx < len(metadata_store_farmer)
#     ]
#     if not valid_docs:
#         return jsonify({"error": "No relevant context found for the query."}), 400

#     # Step 5: Use the context to generate an answer
#     context = " ".join(valid_docs)
#     ai_response = generate_answer(rephrased_question, context[:500], "English")

#     return jsonify({"response": ai_response})


def chat_milk_man():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Detect the language of the original message
    detected_lang = detect_language(user_message)
    print(f"Detected language: {detected_lang}")

    # Translate to English if necessary
    translated_message = (
        translate_to_english(user_message)
        if detected_lang != "en"
        else user_message
    )
    print(f"Translated message: {translated_message}")

    # Generate embeddings for the query
    query_embeddings = preprocess_and_vectorize(translated_message)
    if not query_embeddings:
        return jsonify({"error": "Failed to generate embeddings for the query."}), 400

    query_embedding = query_embeddings[0][1]

    # Perform FAISS search
    distances, indices = faiss_index_milk_man.search(np.array([query_embedding]), k=1)

    # Safeguard: Ensure indices are valid
    valid_docs = [
        metadata_store_milk_man[idx]['text']
        for idx in indices[0]
        if 0 <= idx < len(metadata_store_milk_man)
    ]
    if not valid_docs:
        return jsonify({"error": "No relevant context found for the query."}), 400

    # Use the context to generate an answer
    context = " ".join(valid_docs)
    ai_response = generate_answer(translated_message, context[:500], "English")

    # Translate response back if necessary
    if detected_lang != "en":
        ai_response = GoogleTranslator(source="en", target=detected_lang).translate(ai_response)

    return jsonify({"response": ai_response})


if __name__ == '__main__':
    app.run(debug=True)
