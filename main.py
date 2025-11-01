# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Import LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load API keys from environment (Render will provide these)
load_dotenv() 

# --- Setup ---
# This setup code will run once when your API server starts on Render

PINECONE_INDEX_NAME = "rag-framework" # Your index name

# 1. Initialize Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Initialize Vector Store (the "Retriever")
# This connects to your existing Pinecone index
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# 3. Initialize LLM (The "Thinker")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 4. Create a Prompt Template
template = """
You are an assistant. Answer the user's question based *only* on the following context.
If you don't know the answer from the context, say 'I am not sure.'

Context:
{context}

Question:
{question}
"""
prompt = PromptTemplate.from_template(template)

# 5. Create the RAG "Chain"
# This defines the flow of data for every request

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- API ---

app = FastAPI()

# This defines the JSON body your app must send
class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"Status": "API is running!"}

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    # .invoke() runs the entire chain with the user's question
    try:
        response = rag_chain.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}
