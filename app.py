import os
import json
import hashlib
import numpy as np
import faiss
import re
import torch
import streamlit as st

from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# CONFIG
# =========================
MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-large"
STORE_DIR = "store"
USE_GPU = True

HNSW_M = 64
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# =========================
# HELPERS
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def stable_doc_id(name):
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{base}-{h}"

def read_pdf_text(pdf):
    reader = PdfReader(pdf)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def build_index(emb):
    dim = emb.shape[1]
    idx = faiss.IndexHNSWFlat(dim, HNSW_M)
    idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    return faiss.IndexIDMap2(idx)

# =========================
# ANALYZER
# =========================
class PDFAnalyzer:
    def __init__(self):
        self.device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.reranker = CrossEncoder(RERANK_MODEL)
        ensure_dir(STORE_DIR)

    def index_pdf(self, pdf):
        doc_id = stable_doc_id(pdf.name)

        emb_p = f"{STORE_DIR}/{doc_id}.emb.npy"
        sent_p = f"{STORE_DIR}/{doc_id}.sent.json"
        idx_p = f"{STORE_DIR}/{doc_id}.index"

        if os.path.exists(emb_p):
            st.session_state["meta"] = {
                "sentences": json.load(open(sent_p)),
                "embeddings": np.load(emb_p),
                "index": faiss.read_index(idx_p)
            }
            return

        text = read_pdf_text(pdf)
        sentences = split_sentences(text)

        emb = self.model.encode(
            sentences,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32
        ).astype("float32")

        faiss.normalize_L2(emb)
        index = build_index(emb)
        index.add_with_ids(emb, np.arange(len(sentences)))

        np.save(emb_p, emb)
        json.dump(sentences, open(sent_p, "w"))
        faiss.write_index(index, idx_p)

        st.session_state["meta"] = {
            "sentences": sentences,
            "embeddings": emb,
            "index": index
        }

    # =========================
    # AUTOMATIC EFFICIENT ANSWER
    # =========================
    def efficient_answer(self, question, max_facts=12):
        meta = st.session_state.get("meta")
        if not meta:
            return "Please upload a PDF first."

        index = meta["index"]
        sents = meta["sentences"]

        index.index.hnsw.efSearch = HNSW_EF_SEARCH
        q_emb = self.model.encode([question], normalize_embeddings=True).astype("float32")

        _, ids = index.search(q_emb, min(max_facts * 3, index.ntotal))
        candidates = [sents[i] for i in ids[0] if i != -1]

        scores = self.reranker.predict([(question, c) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # Compression & deduplication
        output = []
        for sent, _ in ranked:
            sent = sent.strip()
            if all(sent.lower() not in o.lower() for o in output):
                sent = sent.replace("The ", "").replace("This ", "")
                output.append(sent)
            if len(output) == 6:
                break

        return "\n".join(f"- {o}" for o in output)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Efficient PDF Answering", layout="centered")
st.title("📘 Efficient PDF Answering System")

analyzer = PDFAnalyzer()

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    with st.spinner("Indexing document..."):
        analyzer.index_pdf(uploaded)
        st.success("PDF indexed successfully")

question = st.text_input("Ask a question (exam style)")

if question and "meta" in st.session_state:
    with st.spinner("Generating efficient answer..."):
        answer = analyzer.efficient_answer(question)
        st.markdown("### ✅ Efficient Answer")
        st.markdown(answer)
