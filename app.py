import os
import json
import hashlib
import time
import numpy as np
import faiss
import re
import torch
import streamlit as st
from typing import List, Dict, Tuple, Optional
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass

# =========================
# CONFIGURATION (Lightweight)
# =========================
@dataclass
class Config:
    MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    RERANK_MODEL: str = "BAAI/bge-reranker-large"
    STORE_DIR: str = "store"
    CHUNK_SIZE: int = 500
    MAX_RESULTS: int = 10

# =========================
# STREAMLIT-SAFE HELPERS
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def stable_doc_id(name):
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r'[^\w\-]', '_', base)[:50]
    h = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{base}_{h}"

def validate_pdf(pdf_file):
    """Quick PDF validation"""
    try:
        pdf_file.seek(0, 2)
        size = pdf_file.tell()
        pdf_file.seek(0)
        
        if size == 0:
            return False, "File is empty"
        if size > 50 * 1024 * 1024:  # 50MB
            return False, f"File too large ({size/1024/1024:.1f}MB)"
        
        # Quick text extraction test
        reader = PdfReader(pdf_file)
        if len(reader.pages) == 0:
            return False, "PDF has no pages"
        
        first_page = reader.pages[0].extract_text() or ""
        if len(first_page.strip()) < 10:
            return False, "PDF may be scanned or unreadable"
            
        return True, "Valid PDF"
    except Exception as e:
        return False, f"Error: {str(e)}"

def split_text_smart(text, chunk_size=500, chunk_overlap=50):
    """Streamlit-safe text splitting"""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_length = len(sentence)
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Keep overlap
            if chunk_overlap > 0 and len(current_chunk) > 1:
                current_chunk = current_chunk[-1:]  # Keep last sentence
                current_length = len(current_chunk[0]) if current_chunk else 0
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sent_length + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# =========================
# MODEL MANAGEMENT (Streamlit-safe)
# =========================
@st.cache_resource
def load_models():
    """Load models once and cache them - CRITICAL for Streamlit"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner("Loading models (first time only)..."):
        encoder = SentenceTransformer(Config.MODEL_NAME, device=device)
        reranker = CrossEncoder(Config.RERANK_MODEL, device=device)
    
    return encoder, reranker

# =========================
# INDEX MANAGEMENT
# =========================
class StreamlitSafeIndexer:
    """Minimal indexer that avoids state issues"""
    
    @staticmethod
    def build_index(embeddings):
        """Build FAISS index"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Simple and stable
        index.add(embeddings)
        return index
    
    @staticmethod
    def save_index(index, path):
        """Save index to disk"""
        faiss.write_index(index, path)
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_index(_doc_id):
        """Load index from disk with caching"""
        path = f"{Config.STORE_DIR}/{_doc_id}.index"
        if os.path.exists(path):
            return faiss.read_index(path)
        return None

# =========================
# CORE FUNCTIONS (Stateless)
# =========================
def extract_pdf_text(pdf_file):
    """Extract text from PDF"""
    reader = PdfReader(pdf_file)
    text_parts = []
    
    progress_bar = st.progress(0)
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text.strip())
        progress_bar.progress((i + 1) / len(reader.pages))
    
    progress_bar.empty()
    return "\n".join(text_parts)

def process_pdf(pdf_file, encoder):
    """Process PDF and create embeddings"""
    # Generate document ID
    doc_id = stable_doc_id(pdf_file.name)
    
    # Check if already processed
    emb_path = f"{Config.STORE_DIR}/{doc_id}.emb.npy"
    text_path = f"{Config.STORE_DIR}/{doc_id}.text.json"
    index_path = f"{Config.STORE_DIR}/{doc_id}.index"
    
    if os.path.exists(emb_path) and os.path.exists(index_path):
        # Load from cache
        with st.spinner("Loading cached embeddings..."):
            embeddings = np.load(emb_path)
            chunks = json.load(open(text_path, 'r'))
            index = faiss.read_index(index_path)
        
        st.success(f"Loaded cached document: {pdf_file.name}")
        return doc_id, chunks, embeddings, index
    
    # Process new PDF
    with st.spinner("Extracting text from PDF..."):
        text = extract_pdf_text(pdf_file)
        if not text.strip():
            st.error("No text could be extracted from PDF")
            return None, None, None, None
    
    with st.spinner("Splitting text into chunks..."):
        chunks = split_text_smart(text, Config.CHUNK_SIZE, 50)
        if not chunks:
            st.error("Failed to split text into chunks")
            return None, None, None, None
        
        st.info(f"Created {len(chunks)} text chunks")
    
    with st.spinner("Creating embeddings (this may take a moment)..."):
        # Process in batches to show progress
        batch_size = 16
        all_embeddings = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_emb = encoder.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_emb)
            
            # Update progress
            progress = min((i + batch_size) / len(chunks), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {min(i + batch_size, len(chunks))} of {len(chunks)}")
        
        progress_bar.empty()
        status_text.empty()
        
        embeddings = np.vstack(all_embeddings).astype("float32")
    
    with st.spinner("Building search index..."):
        index = StreamlitSafeIndexer.build_index(embeddings)
    
    # Save to disk
    with st.spinner("Saving to cache..."):
        ensure_dir(Config.STORE_DIR)
        np.save(emb_path, embeddings)
        with open(text_path, 'w') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        StreamlitSafeIndexer.save_index(index, index_path)
    
    st.success(f"Successfully processed: {pdf_file.name}")
    return doc_id, chunks, embeddings, index

def search_document(question, encoder, reranker, chunks, index, max_results=10):
    """Search document and return answer"""
    if not chunks or index is None:
        return "No document loaded or indexed.", []
    
    # Encode question
    with st.spinner("Encoding question..."):
        q_embedding = encoder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
    
    # Search
    with st.spinner("Searching document..."):
        k = min(max_results * 3, index.ntotal)
        distances, indices = index.search(q_embedding, k)
        
        # Get candidate chunks
        candidates = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(chunks):
                candidates.append(chunks[idx])
    
    if not candidates:
        return "No relevant information found.", []
    
    # Rerank
    with st.spinner("Refining results..."):
        pairs = [(question, chunk) for chunk in candidates]
        scores = reranker.predict(pairs)
        
        # Combine and sort
        results = list(zip(candidates, scores))
        results.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate and format
    seen = set()
    unique_results = []
    for chunk, score in results:
        # Simple deduplication
        chunk_key = chunk[:100].lower()
        if chunk_key not in seen:
            seen.add(chunk_key)
            unique_results.append((chunk, score))
            if len(unique_results) >= max_results:
                break
    
    # Format answer
    if not unique_results:
        return "No unique information found after filtering.", []
    
    answer = "**Key Information:**\n\n"
    for i, (chunk, score) in enumerate(unique_results, 1):
        # Clean up chunk
        clean_chunk = chunk.strip()
        clean_chunk = re.sub(r'^[Tt]he\s+', '', clean_chunk)
        clean_chunk = re.sub(r'^[Tt]his\s+', '', clean_chunk)
        
        answer += f"{i}. {clean_chunk}\n"
    
    return answer, unique_results

# =========================
# STREAMLIT UI (Simplified)
# =========================
def main():
    """Main Streamlit app - clean and simple"""
    # Page config
    st.set_page_config(
        page_title="PDF Q&A Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("📚 PDF Question Answering Assistant")
    st.markdown("Upload a PDF and ask questions about its content")
    
    # Load models (cached)
    encoder, reranker = load_models()
    
    # Initialize session state
    if 'doc_id' not in st.session_state:
        st.session_state.doc_id = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'index' not in st.session_state:
        st.session_state.index = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file is not None:
            # Validate
            is_valid, message = validate_pdf(uploaded_file)
            
            if not is_valid:
                st.error(f"❌ {message}")
            else:
                # Check if same file
                if st.session_state.current_file != uploaded_file.name:
                    st.session_state.current_file = uploaded_file.name
                    
                    # Process PDF
                    with st.spinner("Processing PDF..."):
                        doc_id, chunks, embeddings, index = process_pdf(
                            uploaded_file, encoder
                        )
                        
                        if doc_id:
                            st.session_state.doc_id = doc_id
                            st.session_state.chunks = chunks
                            st.session_state.index = index
                            st.success(f"✅ Ready: {uploaded_file.name}")
                        else:
                            st.error("Failed to process PDF")
        
        st.divider()
        
        st.header("⚙️ Settings")
        
        Config.MAX_RESULTS = st.slider(
            "Results to show",
            min_value=3,
            max_value=20,
            value=10,
            help="Number of results to display"
        )
        
        Config.CHUNK_SIZE = st.slider(
            "Chunk size",
            min_value=200,
            max_value=1000,
            value=500,
            step=50,
            help="Size of text chunks (characters)"
        )
        
        if st.button("Clear Cache", type="secondary"):
            # Clear session state but keep models
            keys = ['doc_id', 'chunks', 'index', 'current_file']
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.divider()
        
        st.caption(f"Model: {Config.MODEL_NAME}")
        st.caption(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        if st.session_state.chunks is None:
            st.info("👈 Please upload a PDF file to get started")
        else:
            question = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the main findings?",
                height=100
            )
            
            if question:
                if st.button("Get Answer", type="primary"):
                    with st.spinner("Searching for answer..."):
                        answer, results = search_document(
                            question,
                            encoder,
                            reranker,
                            st.session_state.chunks,
                            st.session_state.index,
                            Config.MAX_RESULTS
                        )
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show details in expander
                    with st.expander("Show details and scores"):
                        for i, (chunk, score) in enumerate(results, 1):
                            st.markdown(f"**Result {i}** (Score: {score:.3f}):")
                            st.caption(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                            st.divider()
    
    with col2:
        st.header("📊 Document Info")
        
        if st.session_state.doc_id:
            st.success("✅ Document Loaded")
            
            # Show stats
            if st.session_state.chunks:
                chunks = st.session_state.chunks
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Chunks", len(chunks))
                with col_stat2:
                    avg_len = np.mean([len(c) for c in chunks])
                    st.metric("Avg Length", f"{avg_len:.0f} chars")
                
                # Sample chunks
                with st.expander("View sample chunks"):
                    for i, chunk in enumerate(chunks[:3], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(chunk[:150] + "..." if len(chunk) > 150 else chunk)
                        st.divider()
            
            # Document metadata
            with st.expander("Document Metadata"):
                st.write(f"Document ID: `{st.session_state.doc_id}`")
                if st.session_state.current_file:
                    st.write(f"File: `{st.session_state.current_file}`")
                
                if st.session_state.index:
                    st.write(f"Index size: {st.session_state.index.ntotal} vectors")
        
        else:
            st.info("No document loaded")
    
    # Footer
    st.divider()
    st.caption("Powered by BGE embeddings and FAISS search")

if __name__ == "__main__":
    main()
