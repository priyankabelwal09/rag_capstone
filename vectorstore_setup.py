import os
#os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"  # <-- Important line
import streamlit as st
import pickle
from groq import Groq
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceInstructorEmbeddings
#from sentence_transformers import SentenceTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

#file_path="faiss_index"
load_dotenv()
# Load PDF
loaders = [
    PyPDFLoader("OpenStack_RAG_Context_Document.pdf"),  
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

#print(len(docs))
# Initializing the RecursiveCharacterTextSplitter with chunk size and overlap parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,       # The maximum size of each chunk (500 characters)
    chunk_overlap = 50      # The number of characters that will overlap between consecutive chunks (50 characters)
)
# Splitting the documents into smaller chunks using the defined text_splitter
splits = text_splitter.split_documents(docs)

# Printing the number of splits (chunks) generated
print(len(splits))

# Printing the length of the content of the first chunk
print(len(splits[0].page_content))

#embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#embedding=HuggingFaceInstructorEmbeddings()
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
vectordb= FAISS.from_documents(docs,embedding)
vectordb.save_local("faiss_index")

#with open(file_path, "wb") as f:
    #pickle.dump(vectordb, f)

print("Vector store created and saved to")