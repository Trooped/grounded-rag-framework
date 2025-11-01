# main.py
import os
import json
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

# LangChain / OpenAI / Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment (Render provides these)
load_dotenv()

# --- API Security (for /upload endpoint) ---
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
auth_scheme = HTTPBearer()

def check_api_key(creds: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not creds or creds.scheme != "Bearer" or creds.credentials != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Admin API key",
        )
    return True

# --- Global Objects (Load once, reuse) ---
print("--- Loading heavy models (once on startup) ---")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)  # streaming enabled

# Multi-lingual Chat Prompt
template = (
    "You are a helpful assistant. Answer the user's question based *only* on the following context.\n"
    "Respond in the *same language* as the user's question.\n"
    "Be polite and helpful.\n"
    "If you do not know the answer from the context, say so in the user's language.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer (in the same language as the question):"
)
prompt = ChatPromptTemplate.from_template(template)

PINECONE_INDEX_NAME = "rag-framework"

# --- API ---
app = FastAPI()

# --- Helper Function for Ingestion (Async) ---
async def ingest_url(url: str, namespace: str):
    try:
        print(f"Ingesting {url} into namespace {namespace}...")
        loader = WebBaseLoader(url)
        # Prefer async load if available; otherwise use a thread
        try:
            docs = await loader.aload()
        except AttributeError:
            docs = await asyncio.to_thread(loader.load)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        await PineconeVectorStore.afrom_documents(
            splits,
            embeddings,
            index_name=PINECONE_INDEX_NAME,
            namespace=namespace,
        )
        print(f"Successfully ingested {url}.")
        return True
    except Exception as e:
        print(f"Error ingesting {url}: {e}")
        return False

# --- Pydantic Models (Data Contracts) ---
class ChatRequest(BaseModel):
    question: str
    namespace: str  # e.g., "betvuna" or "other_org"

class UploadRequest(BaseModel):
    url: str
    namespace: str  # e.g., "betvuna"

# --- Utilities ---
def _join_docs(docs):
    """Turn retrieved docs into a readable context block (preserving sources)."""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] {src}\n{d.page_content}")
    return "\n\n".join(parts)

# --- Streaming Chat (SSE) ---
async def _sse_chat_stream(req: ChatRequest):
    """
    Async generator that yields SSE events.
    - Each chunk is framed as NDJSON in the SSE 'data:' field: {"token": "..."}
    - Emits a final 'done' event so clients can close cleanly.
    """
    try:
        print(f"Streaming chat request for namespace: {req.namespace}")

        # Vector store + retriever for the given namespace
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=req.namespace,
        )
        retriever = vectorstore.as_retriever()

        # RAG chain:
        #   input -> {"context": retriever(input), "question": input}
        #   -> join docs -> prompt -> llm -> parse to str
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(lambda x: {
                "context": _join_docs(x["context"]),
                "question": x["question"],
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        # Stream text tokens exactly as produced (spaces preserved)
        async for token in rag_chain.astream(req.question):
            yield {"data": json.dumps({"token": token}, ensure_ascii=False)}
        # Signal completion explicitly
        yield {"event": "done", "data": "{}"}
    except Exception as e:
        # Send an error event and end the stream
        yield {"event": "error", "data": json.dumps({"error": str(e)})}
    # When the generator returns, EventSourceResponse closes the connection.

# --- Endpoints ---
@app.get("/")
async def read_root():
    return {"status": "RAG API is running!"}

@app.post("/chat")
async def chat_with_rag(request: ChatRequest):
    """
    SSE streaming endpoint. In Postman, set:
      - Headers: Accept: text/event-stream
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # helps bypass proxy buffering (nginx, cloudflare)
    }
    return EventSourceResponse(_sse_chat_stream(request), headers=headers)

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
