import os
#os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"  # <-- Important line
import streamlit as st
#import pickle
from groq import Groq
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

#file_path="faiss_index.pkl"

#def load_vectorstore():
    #if os.path.exists(file_path):
        #with open(file_path, "rb") as f:
    # Load the vector store from the pickle file
           # return pickle.load(f)
           


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
groq = Groq()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# Load vector DB
vectorstore = FAISS.load_local("faiss_index", embedding,allow_dangerous_deserialization=True)
# === Query with LLaMA 3.3 70B from Groq ===
def get_llama_response(prompt: str) -> str:
    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === RAG Pipeline ===
def get_rag_answer(query: str, vectorstore) -> str:
    relevant_docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""You are an OpenStack troubleshooting assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    return get_llama_response(prompt)


# === Streamlit UI ===
st.set_page_config(page_title="OpenStack Chatbot", layout="wide")
st.title("OpenStack Chatbot")
st.markdown("Ask anything based on the OpenStack logs.")

#vectorstore = load_vectorstore()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask a question...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_rag_answer(user_input, vectorstore)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


