from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import RAGDemo
from config import GEMINI_API_KEY, CORS_ORIGINS

app = FastAPI(title="Legal RAG API")

frontend_url_regex = r"https://legal-rag-demo.*\.vercel\.app|http://localhost:3000"
# CORS middleware - allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=frontend_url_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system = RAGDemo(gemini_api_key=GEMINI_API_KEY)

# Preload GDPR embeddings on startup
@app.on_event("startup")
async def startup_event():
    """Preload GDPR embeddings when the app starts"""
    try:
        print("\n" + "=" * 70)
        print("STARTUP: Preloading GDPR embeddings...")
        print("=" * 70)
        
        gdpr_path = "example_data/gdpr.pdf"
        embeddings_path = "example_data/gdpr_embeddings.pkl"
        
        if os.path.exists(gdpr_path):
            rag_system.setup(
                pdf_path=gdpr_path,
                use_precomputed=True,
                precomputed_embedding_path=embeddings_path
            )
            print("✓ GDPR embeddings preloaded and ready")
        else:
            print(f"⚠ Warning: GDPR file not found at {gdpr_path}")
            print("  GDPR will be loaded on first request")
        
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"⚠ Warning: Could not preload GDPR: {str(e)}")
        print("  GDPR will be loaded on first request\n")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class DocumentInfo(BaseModel):
    name: str
    chunks: int

@app.get("/")
def root():
    return {"message": "Legal RAG API", "status": "running"}

@app.post("/api/load-gdpr")
async def load_gdpr():
    """Load pre-loaded GDPR document"""
    global rag_system
    
    try:
        # Setup with precomputed embeddings
        rag_system.setup(
            pdf_path="example_data/gdpr.pdf",
            use_precomputed=True,
            precomputed_embedding_path="example_data/gdpr_embeddings.pkl"
        )
        chunks = len(rag_system.vector_store.chunks)
        
        return {
            "success": True,
            "document": "gdpr.pdf",
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process custom document"""
    global rag_system
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with RAG system (no precomputed embeddings for uploads)
        rag_system.setup(
            pdf_path=temp_path,
            use_precomputed=False,
            precomputed_embedding_path=None
        )
        chunks = len(rag_system.vector_store.chunks)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "success": True,
            "document": file.filename,
            "chunks": chunks
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask question about loaded document"""
    global rag_system
    
    if not rag_system or not rag_system.vector_store.chunks:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    try:
        result = rag_system.answer(
            request.question,
            top_k=request.top_k,
        )

        return {
            "answer": result["answer"],
            "chunks": result["chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "document_loaded": len(rag_system.vector_store.chunks) > 0 if rag_system else False,
        "num_chunks": len(rag_system.vector_store.chunks) if rag_system else 0
    }
