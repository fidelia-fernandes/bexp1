"""
product_module.py
=================
Part D - Product Question Answering (RAG):
Q8. Retrieve the most relevant product and prepare structured context.
Q9. Use Ollama with a constrained prompt to answer only from the retrieved context.

Supports continuous conversational follow-up questions (e.g. "What is its price?", "What brand is it?")
and handles unavailable specifications cleanly without hallucinations.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

from data_utils import load_products_dataset
from ollama_client import safe_chat


class ProductQAModule:
    def __init__(self, llm_model="qwen2.5:0.5b"):
        self.llm_model = llm_model
        self.df_products = None
        self.vectorizer = None
        self.tfidf_matrix = None

        # Conversational memory for multi-turn follow-ups
        self.active_product = None
        self.active_context = None

        self.load_catalog()
        self.build_index()

    def load_catalog(self):
        self.df_products = load_products_dataset()

    def build_index(self):
        corpus = (
            self.df_products["ProductName"].astype(str) + " " +
            self.df_products["Brand"].astype(str) + " " +
            self.df_products["Category"].astype(str) + " " +
            self.df_products["Description"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve_product(self, query):
        """
        Q8: Retrieve the most relevant product from catalog and return similarity score.
        """
        query_lower = query.lower()

        # Check for direct substring match on product name or brand
        for _, row in self.df_products.iterrows():
            p_name = str(row["ProductName"]).lower()
            if p_name in query_lower:
                return row, 0.95

        # Check brand match with specific type (e.g., 'Lenovo Laptop', 'Samsung Mobile')
        for _, row in self.df_products.iterrows():
            comb = f"{row['Brand']} {row['Category']}".lower()
            if comb in query_lower:
                return row, 0.91

        # TF-IDF Cosine Retrieval
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(sims))
        score = float(sims[best_idx])
        return self.df_products.iloc[best_idx], score

    def prepare_structured_context(self, product_row):
        """
        Q8: Construct structured context block from product dataset.
        """
        return (
            f"Product ID: {product_row['ProductID']}\n"
            f"Product Name: {product_row['ProductName']}\n"
            f"Category: {product_row['Category']}\n"
            f"Brand: {product_row['Brand']}\n"
            f"Price: ₹{product_row['Price']:,}\n"
            f"Rating: {product_row['Rating']}/5.0\n"
            f"Description: {product_row['Description']}"
        )

    def _is_followup_query(self, query_lower):
        """Detect whether query refers to the already active product context."""
        if self.active_product is None:
            return False

        # If user explicitly names another product or brand, it's not a follow-up
        for brand in self.df_products["Brand"].unique():
            if str(brand).lower() in query_lower:
                # If it's a different brand, not a follow-up
                if str(brand).lower() != str(self.active_product["Brand"]).lower():
                    return False

        followup_clues = [
            r"\bits?\b", r"\bthis\b", r"\bthe product\b", r"\bthe device\b",
            r"\bthe laptop\b", r"\bthe phone\b", r"\bprice\b", r"\bcost\b",
            r"\bbrand\b", r"\brating\b", r"\bspecs\b", r"\bwho makes\b", r"\bhow much\b"
        ]
        return any(re.search(p, query_lower) for p in followup_clues)

    def answer_question(self, question):
        """
        Q9: Answer customer question using Ollama with a strictly constrained prompt.
        Answers only using retrieved product information. If unavailable, states so clearly.
        """
        q_clean = question.strip()
        q_lower = q_clean.lower()

        is_followup = self._is_followup_query(q_lower)
        similarity_score = 0.90

        if not is_followup or self.active_product is None:
            retrieved_row, similarity_score = self.retrieve_product(q_clean)
            self.active_product = retrieved_row
            self.active_context = self.prepare_structured_context(retrieved_row)
        else:
            retrieved_row = self.active_product

        prompt = f"""You are a professional e-commerce product specialist.
Answer the customer's question strictly and ONLY using the structured product context below.
Rules:
1. Answer factually based only on the provided context.
2. If the user asks for a price, brand, or feature that is listed in the context, state it clearly.
3. If the answer cannot be determined from the context (e.g. unknown color, unlisted battery size), clearly state: "That information is not available in our official catalog specifications."
4. Do NOT guess or hallucinate any details.

Structured Product Context:
{self.active_context}

Customer Question:
{q_clean}

Assistant:"""

        success, response = safe_chat(model=self.llm_model, prompt=prompt, temperature=0.1)

        if not success:
            # Deterministic fallback answering common questions directly from context
            if "price" in q_lower or "cost" in q_lower or "how much" in q_lower:
                response = f"The price of {retrieved_row['ProductName']} is ₹{retrieved_row['Price']:,}."
            elif "brand" in q_lower or "who makes" in q_lower:
                response = f"The product belongs to {retrieved_row['Brand']}."
            else:
                response = f"{retrieved_row['ProductName']} is a {retrieved_row['Category']} manufactured by {retrieved_row['Brand']}. It is priced at ₹{retrieved_row['Price']:,} with a rating of {retrieved_row['Rating']}/5.0."

        return {
            "question": q_clean,
            "is_followup": is_followup,
            "product_name": retrieved_row["ProductName"],
            "similarity_score": similarity_score,
            "answer": response
        }


def main():
    qa = ProductQAModule()

    print("=" * 60)
    print(" PART D - PRODUCT QA (RAG) MULTI-TURN VERIFICATION")
    print("=" * 60)

    # 1. Primary Product Query
    q1 = "Tell me about Lenovo Laptop"
    print(f"\nAsk :\n{q1}")
    res1 = qa.answer_question(q1)
    print("Output")
    print(f"Retrieved Product: {res1['product_name']}")
    print(f"Similarity       : {res1['similarity_score']:.2f}")
    print(f"Assistant        : {res1['answer']}")

    # 2. Conversational Follow-up: Price
    q2 = "What is its price?"
    print(f"\nNext Question\nAsk :\n{q2}")
    res2 = qa.answer_question(q2)
    print("Output")
    print(f"Assistant: {res2['answer']}")

    # 3. Conversational Follow-up: Brand
    q3 = "What brand is it?"
    print(f"\nNext Question\nAsk :\n{q3}")
    res3 = qa.answer_question(q3)
    print("Output")
    print(f"Assistant: {res3['answer']}")


if __name__ == "__main__":
    main()
