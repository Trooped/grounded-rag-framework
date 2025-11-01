# main.py
import os
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

# Import LangChain components
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API keys from environment (Render will provide these)
load_dotenv() 

# --- API Security (for /upload endpoint) ---
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
auth_scheme = HTTPBearer()

def check_api_key(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not creds or creds.scheme != "Bearer" or creds.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Admin API key"
        )
    return True

# --- Global Objects (Load once, reuse) ---
print("--- Loading heavy models (once on startup) ---")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True) # Enable streaming

# Multi-lingual Prompt Template
template = """
You are a helpful assistant. Answer the user's question based *only* on the following context.
Respond in the *same language* as the user's question.
Be polite and helpful.
If you do not know the answer from the context, say so in the user's language. 
(For example: if the question is in Hebrew, say 'איני בטוח לגבי זה'. If in English, say 'I am not sure about that'.)

Context:
{context}

Question:
{question}

Answer (in the same language as the question):
"""
prompt = PromptTemplate.from_template(template)

PINECONE_INDEX_NAME = "rag-framework" 

# --- API ---
app = FastAPI()

# --- Helper Function for Ingestion (Async) ---
async def ingest_url(url: str, namespace: str):
    try:
        print(f"Ingesting {url} into namespace {namespace}...")
        loader = WebBaseLoader(url)
        docs = await loader.aload() # Async load

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        await PineconeVectorStore.afrom_documents(
            splits, 
            embeddings, 
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace
        )
        print(f"Successfully ingested {url}.")
        return True
    except Exception as e:
        print(f"Error ingesting {url}: {e}")
        return False

# --- Pydantic Models (Data Contracts) ---
class ChatRequest(BaseModel):
    question: str
    namespace: str # e.g., "betvuna" or "other_org"

class UploadRequest(BaseModel):
    url: str
    namespace: str # e.g., "betvuna"

# --- Streaming Chat Generator ---
async def stream_chat(request: ChatRequest):
    """
    Async generator for streaming responses.
    This keeps the connection alive and bypasses timeouts.
    """
    try:
        print(f"Streaming chat request for namespace: {request.namespace}")

        # 1. Create a VectorStore instance pointing to the *specific namespace*
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings,
            namespace=request.namespace
        )

        # 2. Create a retriever for that namespace
        retriever = vectorstore.as_retriever()

        # 3. Build the RAG chain for this request
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 4. Use .astream() to get chunks as they are generated
        async for chunk in rag_chain.astream(request.question):
            if chunk:
                yield {"data": chunk}

    except Exception as e:
        print(f"Error during RAG stream: {e}")
        # Also send the error in the correct format
        yield {"data": f"Error: {str(e)}"}

# --- Endpoints ---
@app.get("/")
async def read_root():
    return {"Status": "RAG API is running!"}

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """
    Main chat endpoint. Receives a question and a namespace,
    and returns a StreamingResponse.
    """
    return EventSourceResponse(stream_chat(request))

@app.post("/upload", dependencies=[Depends(check_api_key)])
async def upload_data(request: UploadRequest):
    """
    Secured endpoint for an admin to add new knowledge.
    Requires a Bearer token in the Authorization header.
    """
    success = await ingest_url(request.url, request.namespace)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to ingest URL.")

    return {"status": "success", "message": f"URL {request.url} ingested into namespace {request.namespace}."}

