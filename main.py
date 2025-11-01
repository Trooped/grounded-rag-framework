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

# --- Global Cache ---
# We initialize this as None. It will be "lazy loaded"
# the first time the /chat endpoint is called.
rag_chain = None

def get_rag_chain():
    """
    Initializes and returns the RAG chain.
    Uses a global variable to cache the chain so we don't
    re-initialize it on every single request.
    """
    global rag_chain

    # If the chain is already initialized, just return it.
    if rag_chain:
        return rag_chain

    # --- First-time setup (will be slow on first request) ---
    print("--- Initializing RAG chain (first request only) ---")

    PINECONE_INDEX_NAME = "rag-framework" 

    # 1. Initialize Embedding Model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Initialize Vector Store (the "Retriever")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # 3. Initialize LLM
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

    # 5. Create and cache the RAG "Chain"
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("--- RAG chain initialized successfully ---")
    return rag_chain

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
        # This will initialize the chain on the first call,
        # and re-use the cached chain on subsequent calls.
        chain = get_rag_chain()
        response = chain.invoke(request.question)
        return {"answer": response}
    except Exception as e:
        # This will catch the *real* error (like "index not found")
        # and return it as a JSON, so we can debug it.
        print(f"Error during RAG chain invocation: {e}")
        return {"answer": f"An error occurred: {str(e)}"}
