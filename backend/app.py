## -------------------- Imports -------------------- ##
import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import traceback, logging
import requests
from pydantic import BaseModel
import pandas as pd

from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from functions import chunk_data, calculate_embedding_cost, create_embeddings


## -------------------- Load Env & CSV -------------------- ##
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

csv_path = "/Users/katerinatelegina/Library/Mobile Documents/com~apple~CloudDocs/RAG-bookchat/RAG-Chatbot/backend/books.csv"
file_name = os.path.basename(csv_path)
df = pd.read_csv(csv_path)

# — Normalize Popularity into booleans, just like in the frontend —
df["Popularity"] = (
    df["Popularity"]
      .astype(str)
      .str.lower()
      .map({"checked": True, "true": True})
      .fillna(False)
      .astype(bool)
)

# Turn each row into a LangChain Document
documents = [
    Document(
        page_content=row["Description"],
        metadata={
            "source": file_name,
            "title": row["Title"],
            "author": row["Author"],
            "popularity": row["Popularity"],
            "genre": row["Genre"],
            "rating": row["Rating"],
        },
    )
    for _, row in df.iterrows()
]

## -------------------- FastAPI App & Vector Store -------------------- ##
app = FastAPI()
vector_store = None  # will be set in startup

@app.on_event("startup")
async def startup_event():
    global vector_store
    chunks = chunk_data(documents)
    total_tokens, cost = calculate_embedding_cost(chunks)
    vector_store = create_embeddings(chunks)
    print(f"✅ Vector store ready: {len(chunks)} chunks ⸺ cost ${cost:.4f}")

## -------------------- Request Schema -------------------- ##
class QueryRequest(BaseModel):
    query: str

## -------------------- Recommendation Endpoint -------------------- ##
@app.post("/generate-response/")
async def generate_response(req: QueryRequest):
    if vector_store is None:
        logging.error("Vector store was None at request time!")
        raise HTTPException(status_code=500, detail="Vector store not initialized.")

    # 1) Build your system prompt and retriever
    system_prompt = (
        "You are a smart book recommender. "
        "You are original, so never recommend the same book tweice or the book that you know is popular."
        "Use the vector store to find relevant descriptions and answer the user’s prompt. "
        "When asked to recommend a similar book, return 3–6 items in this format:\n"
        "# Title: ...\n"
        "# Author: ...\n"
        "# Genre: ...\n"
        "# Summary: ...\n"
        "# Rating: ★★★★☆\n"
        "Include the star rating based on the metadata."
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "filter": {"popularity": False}  # now works!
        }
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(
            "Here are the relevant excerpts:\n\n"
            "{summaries}\n\n"
            "User asks: {question}"
        )
    ])

    try:
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": chat_prompt},
        )
        result = chain({"question": req.query})
    except Exception as e:
        # Log full stack to your console
        logging.error("Error in /generate-response:\n" + traceback.format_exc())
        # Return the exception message in JSON
        raise HTTPException(status_code=500, detail=str(e))

    answer = re.sub(r"\s*SOURCES:$", "", result["answer"].strip())
    sources = result.get("sources", "")
    return {"answer": answer, "sources": sources}