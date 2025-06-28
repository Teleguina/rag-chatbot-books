# backend/functions.py

import tiktoken
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(data)

def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum(len(enc.encode(doc.page_content)) for doc in texts)
    cost_usd = total_tokens / 1000 * 0.0004
    return total_tokens, cost_usd

def create_embeddings(chunks):
    embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(chunks, embedder)
