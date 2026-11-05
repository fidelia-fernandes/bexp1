"""
faq_module.py
=============
Part B - FAQ Semantic Search:
Q4. Retrieve the best FAQ using cosine similarity.
Q5. Apply similarity threshold. If matched, use Ollama to generate a natural response;
    otherwise continue to the next module.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
from ollama_client import safe_embeddings, safe_chat


class FAQModule:
    def __init__(self, threshold=0.50, embed_model="nomic-embed-text", llm_model="qwen2.5:0.5b"):
        self.threshold = threshold
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.df_faqs = None
        self.embeddings = None
        self.tfidf_data = None
        self.load_resources()

    def _locate_file(self, filename):
        candidates = [
            os.path.join(BASE_DIR, "models", filename),
            os.path.join(BASE_DIR, "ECommerceGPT", "models", filename),
            os.path.join(BASE_DIR, filename),
            filename
        ]
        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)
        return None

    def load_resources(self):
        # 1. Load FAQs
        self.df_faqs = load_faqs_dataset()

        # 2. Load Embeddings
        emb_path = self._locate_file("faq_embeddings.npy")
        if emb_path:
            try:
                self.embeddings = np.load(emb_path)
            except Exception:
                self.embeddings = None

        # 3. Load TF-IDF Fallback
        tfidf_path = self._locate_file("faq_tfidf.pkl")
        if tfidf_path:
            try:
                with open(tfidf_path, "rb") as f:
                    self.tfidf_data = pickle.load(f)
            except Exception:
                self.tfidf_data = None

    def retrieve_best_faq(self, query):
        """
        Q4: Retrieve best matching FAQ using cosine similarity.
        Returns: (best_question, best_answer, similarity_score)
        """
        # Try dense vector embeddings with Ollama
        if self.embeddings is not None:
            ok, user_emb = safe_embeddings(model=self.embed_model, text=query)
            if ok:
                user_vec = np.array(user_emb, dtype=np.float32).reshape(1, -1)
                sims = cosine_similarity(user_vec, self.embeddings)[0]
                best_idx = int(np.argmax(sims))
                score = float(sims[best_idx])
                row = self.df_faqs.iloc[best_idx]
                return row["Question"], row["Answer"], score

        # TF-IDF Fallback
        if self.tfidf_data is not None:
            vectorizer = self.tfidf_data["vectorizer"]
            matrix = self.tfidf_data["matrix"]
            user_vec = vectorizer.transform([query])
            sims = cosine_similarity(user_vec, matrix)[0]
            best_idx = int(np.argmax(sims))
            score = float(sims[best_idx])
            row = self.df_faqs.iloc[best_idx]
            return row["Question"], row["Answer"], score

        # Exact row match fallback
        for _, row in self.df_faqs.iterrows():
            if str(row["Question"]).lower() in query.lower():
                return row["Question"], row["Answer"], 1.0

        row = self.df_faqs.iloc[0]
        return row["Question"], row["Answer"], 0.0

    def generate_natural_response(self, user_query, best_question, official_answer):
        """
        Q5: Use Ollama to generate a natural, conversational response grounded in the official answer.
        """
        prompt = f"""You are a professional e-commerce customer support chatbot.
A customer is asking: "{user_query}"

Our official store policy is:
FAQ Question: "{best_question}"
Official Answer: "{official_answer}"

Provide a natural, helpful, and concise answer directly addressing the customer's question based on the official policy.
Do not invent policies."""

        success, content = safe_chat(model=self.llm_model, prompt=prompt, temperature=0.2)
        if success:
            return content

        # Transparent fallback with diagnostic status
        return f"{content}\n[Store Policy: {official_answer}]"

    def process_query(self, user_query):
        """
        Full FAQ Pipeline:
        1. Retrieve best match with cosine similarity.
        2. Apply similarity threshold.
        3. If matched, generate natural response with Ollama.
        4. Otherwise, flag matched=False to route to next module.
        """
        best_q, best_a, score = self.retrieve_best_faq(user_query)
        matched = score >= self.threshold

        if matched:
            natural_ans = self.generate_natural_response(user_query, best_q, best_a)
            return {
                "matched": True,
                "best_faq": best_q,
                "official_answer": best_a,
                "similarity_score": score,
                "response": natural_ans,
                "next_module": None
            }
        else:
            return {
                "matched": False,
                "best_faq": best_q,
                "official_answer": best_a,
                "similarity_score": score,
                "response": None,
                "next_module": "RECOMMENDATION"
            }


def main():
    print("Loading FAQ Module...")
    faq_mod = FAQModule(threshold=0.50)

    test_queries = [
        "Where is my parcel?",
        "How do I return a product?"
    ]

    for q in test_queries:
        print(f"\nAsk Question : {q}")
        res = faq_mod.process_query(q)
        print(f"Best Matching FAQ : {res['best_faq']}")
        print(f"Similarity Score : {res['similarity_score']:.2f}")
        if res["matched"]:
            print("\nChatbot:")
            print(res["response"])
        else:
            print("\n[Similarity below threshold -> Routing to next module]")


if __name__ == "__main__":
    main()
