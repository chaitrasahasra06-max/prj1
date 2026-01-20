import os
import json
import hashlib
import numpy as np
import faiss
import re
import torch
import streamlit as st
import nltk

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

# Use nltk for faster sentence splitting
nltk.download("punkt", quiet=True)
def split_sentences(text):
    return nltk.sent_tokenize(text)

def build_index(emb):
    dim = emb.shape[1]
    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M)
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH
    return faiss.IndexIDMap2(hnsw)

# =========================
# ANALYZER
# =========================
class PDFAnalyzer:
    def __init__(self):
        self.device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        # Cache models so they don’t reload every run
        self.model, self.reranker = load_models(self.device)
        ensure_dir(STORE_DIR)

    def index_pdf(self, pdf):
        doc_id = stable_doc_id(pdf.name)

        emb_p = f"{STORE_DIR}/{doc_id}.emb.npy"
        sent_p = f"{STORE_DIR}/{doc_id}.sent.json"
        idx_p = f"{STORE_DIR}/{doc_id}.index"

        if os.path.exists(emb_p):
            st.session_state["meta"] = {
                "sentences": json.load(open(sent_p)),
                "embeddings": np.load(emb_p, mmap_mode="r"),
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

    def efficient_answer(self, question, max_facts=12):
        meta = st.session_state.get("meta")
        if not meta:
            return "Please upload a PDF first."

        index, sents = meta["index"], meta["sentences"]

        q_emb = self.model.encode([question], normalize_embeddings=True).astype("float32")
        _, ids = index.search(q_emb, min(max_facts * 5, index.ntotal))
        candidates = [sents[i] for i in ids[0] if i != -1]

        scores = self.reranker.predict([(question, c) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        output, seen = [], set()
        for sent, _ in ranked:
            sent = sent.strip()
            key = sent.lower()
            if key not in seen:
                seen.add(key)
                sent = re.sub(r"^(The|This)\s+", "", sent)
                output.append(sent)
            if len(output) == 6:
                break

        return "\n".join(f"- {o}" for o in output)

# =========================
# STREAMLIT UI
# =========================
@st.cache_resource
def load_models(device="cpu"):
    model = SentenceTransformer(MODEL_NAME, device=device)
    reranker = CrossEncoder(RERANK_MODEL, device=device)
    return model, reranker

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
