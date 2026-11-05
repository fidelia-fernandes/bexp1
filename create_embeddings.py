"""
create_embeddings.py
====================
Part B - Q3: Generate and store FAQ embeddings.
Generates 768-dimensional dense vector embeddings for FAQs using Ollama's nomic-embed-text.
Stores embeddings in models/faq_embeddings.npy and generates TF-IDF representations in models/faq_tfidf.pkl.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_utils import load_faqs_dataset
from ollama_client import safe_embeddings, check_ollama_daemon


def get_models_dir():
    """Ensure models directory exists and return absolute path."""
    candidates = [
        os.path.join(BASE_DIR, "models"),
        os.path.join(BASE_DIR, "ECommerceGPT", "models"),
        "models"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    primary = os.path.abspath(os.path.join(BASE_DIR, "models"))
    os.makedirs(primary, exist_ok=True)
    return primary


def build_and_save_faq_embeddings():
    """
    Q3: Generate and store FAQ embeddings.
    """
    df_faqs = load_faqs_dataset()
    questions = df_faqs["Question"].tolist()
    models_dir = get_models_dir()

    print(f"[create_embeddings] Loaded {len(questions)} FAQ questions.")

    # 1. Ollama Dense Embeddings
    print("[create_embeddings] Generating dense embeddings with Ollama ('nomic-embed-text')...")
    embeddings_list = []
    dense_success = True

    for idx, q in enumerate(questions):
        ok, vec = safe_embeddings(model="nomic-embed-text", text=q)
        if ok:
            embeddings_list.append(vec)
        else:
            print(f"[create_embeddings] Warning on question {idx}: {vec}")
            dense_success = False
            break

    if dense_success and len(embeddings_list) == len(questions):
        emb_arr = np.array(embeddings_list, dtype=np.float32)
        emb_path = os.path.join(models_dir, "faq_embeddings.npy")
        np.save(emb_path, emb_arr)
        print(f"[create_embeddings] Successfully saved dense embeddings to '{emb_path}' (Shape: {emb_arr.shape}).")

    # 2. TF-IDF Fallback Matrix
    print("[create_embeddings] Generating TF-IDF vector index...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    tfidf_path = os.path.join(models_dir, "faq_tfidf.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump({"vectorizer": tfidf_vectorizer, "matrix": tfidf_matrix}, f)
    print(f"[create_embeddings] Successfully saved TF-IDF vectors to '{tfidf_path}'.")

    # 3. Cache processed dataset
    meta_path = os.path.join(models_dir, "faq_data.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(df_faqs, f)
    print(f"[create_embeddings] Successfully cached FAQ dataset to '{meta_path}'.")

    return True


if __name__ == "__main__":
    build_and_save_faq_embeddings()
