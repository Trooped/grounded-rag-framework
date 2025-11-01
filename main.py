# main.py
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

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
# We load the "heavy" models once on startup
print("--- Loading heavy models (once on startup) ---")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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

PINECONE_INDEX_NAME = "rag-framework" # Your index name

# --- API ---
app = FastAPI()

# --- Helper Function for Ingestion ---
def ingest_url(url: str, namespace: str):
    try:
        print(f"Ingesting {url} into namespace {namespace}...")
        loader = WebBaseLoader(url)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        PineconeVectorStore.from_documents(
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
    namespace: str 

class UploadRequest(BaseModel):
    url: str
    namespace: str

# --- Endpoints ---
@app.get("/")
def read_root():
    return {"Status": "RAG API is running!"}

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    """
    Main chat endpoint. Receives a question and a namespace,
    returns a grounded answer.
    """
    try:
        print(f"Chat request for namespace: {request.namespace}")

        # 1. Create a VectorStore instance pointing to the *specific namespace*
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings,
            namespace=request.namespace
        )

        # 2. Create a retriever for that namespace
        retriever = vectorstore.as_retriever()

        # 3. Build the RAG chain for this request
        # We re-build the chain object every time, but the heavy
        # components (llm, prompt, embeddings) are cached globally.
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(request.question)
        return {"answer": response}

    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/upload", dependencies=[Depends(check_api_key)])
def upload_data(request: UploadRequest):
    """
    Secured endpoint for an admin to add new knowledge.
    Requires a Bearer token in the Authorization header.
    """
    success = ingest_url(request.url, request.namespace)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to ingest URL.")

    return {"status": "success", "message": f"URL {request.url} ingested into namespace {request.namespace}."}

