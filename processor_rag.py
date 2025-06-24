from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import streamlit as st
import os
import numpy as np


# --------------------------------------------
# RAG Setup: Imports, embedding model, Chroma
# --------------------------------------------
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "./chroma_db"

# Sample OpenStack logs or knowledge chunks
splits = [
    Document(page_content="2024-06-01 12:42:15.738 ERROR nova.compute.manager Instance failed to spawn: NoValidHost"),
    Document(page_content="VolumeNotFound error in Cinder may indicate deleted or incorrect volume ID"),
    Document(page_content="Neutron DHCP agent can miss subnets if improperly configured"),
]

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()

load_dotenv()
groq = Groq()

def get_rag_answer(query):
    # Retrieve relevant docs
    relevant_docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Create prompt for LLM
    prompt = f"""You are an OpenStack log assistant.
Use the following context to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

    # Call Groq LLaMA 3.3 70B Versatile
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

#query = "Why did Nova fail to spawn the instance?"
query="what is VolumeNotFound error in Cinder"
answer = get_rag_answer(query)
print("RAG Answer:\n", answer)