# Flask & Environment Variables
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# ✅ Updated LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader  # For loading PDFs & directories
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated embeddings import
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # Updated Pinecone import
from langchain_openai import ChatOpenAI  # Updated OpenAI model import

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# ✅ Custom helper functions (make sure these are also updated)
from src.helper import download_hugging_face_embeddings
from src.prompt import *

# Load environment variables
load_dotenv()

# Securely get API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys. Please check your .env file.")

# Initialize Flask
app = Flask(__name__)

# ✅ Load Updated Embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# ✅ Pinecone Vector Store
index_name = "medicalbot-final"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ✅ Create Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Initialize ChatOpenAI with Groq
llm = ChatOpenAI(
    model_name="mixtral-8x7b-32768",
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Define RAG Chain
prompt = PromptTemplate.from_template("Summarize the following: {context}")
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)

    response = rag_chain.invoke({"input": msg})
    
    print("AI Response:", response["answer"])
    return str(response["answer"])

# Run Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)