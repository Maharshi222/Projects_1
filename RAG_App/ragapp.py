import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import streamlit as st


# File handling and database creation
def extract_text_from_files(file_paths: List[str]) -> List[str]:
    """
    Extract text from PDF, CSV, and DOCX files.

    Args:
        file_paths (List[str]): List of file paths.

    Returns:
        List[str]: Extracted text from all files as a list of strings.
    """
    extracted_texts = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = " ".join([page.extract_text() for page in reader.pages])
            extracted_texts.append(text)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            text = " ".join(df.astype(str).stack().tolist())
            extracted_texts.append(text)
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
            extracted_texts.append(text)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    return extracted_texts

def create_faiss_vector_database(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Create a FAISS vector database from a list of texts.

    Args:
        texts (List[str]): List of texts to embed and index.
        model_name (str): Name of the embedding model.

    Returns:
        Tuple[faiss.IndexFlatL2, List[str]]: FAISS index and corresponding texts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=False)
    dimension = embeddings.shape[1]

    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, texts

def query_rag_with_faiss(user_question: str, faiss_index: faiss.IndexFlatL2, texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> str:
    """
    Retrieve relevant context and generate a response using FAISS and RAG.

    Args:
        user_question (str): The user's question.
        faiss_index (faiss.IndexFlatL2): FAISS vector index.
        texts (List[str]): List of indexed texts.
        model_name (str): Name of the embedding model.

    Returns:
        str: Retrieved context and response.
    """
    model = SentenceTransformer(model_name)
    question_embedding = model.encode(user_question, convert_to_tensor=False)
    question_embedding = np.array(question_embedding).astype('float32').reshape(1, -1)

    # Search in FAISS index
    distances, indices = faiss_index.search(question_embedding, k=1)
    best_match_idx = indices[0][0]
    best_match_context = texts[best_match_idx]

    return best_match_context  # You can integrate this into generate_answer if needed.

# Integrate into the chatbot app
st.markdown('<h2 class="sub-header">Upload Files to Build Knowledge Base</h2>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF, CSV, or DOCX files", accept_multiple_files=True, type=["pdf", "csv", "docx"])

# Ensure the 'uploaded_files' directory exists
upload_dir = "uploaded_files"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Save the uploaded files
file_paths = []
for uploaded_file in uploaded_files:
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_paths.append(file_path)

    with st.spinner("🔄 Extracting text and creating FAISS index..."):
        texts = extract_text_from_files(file_paths)
        faiss_index, indexed_texts = create_faiss_vector_database(texts)

        st.success("Knowledge base created successfully!")

    if user_question:
        with st.spinner("🔄 Querying knowledge base..."):
            retrieved_context = query_rag_with_faiss(user_question, faiss_index, indexed_texts)
            answer = generate_answer(user_question, retrieved_context, "Provide a detailed answer.")

        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.markdown(f"**📖 Retrieved Context:** {retrieved_context}")
        st.markdown(f"**💡 Generated Answer:** {answer}")
        st.markdown('</div>', unsafe_allow_html=True)
