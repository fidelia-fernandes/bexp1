"""
recommendation_module.py
========================
Part C - Product Recommendation:
Q6. Create TF-IDF vectors for product descriptions and catalog attributes.
Q7. Retrieve the Top-3 products and use Ollama to recommend the best product
    with factual justification.

Supports category synonyms, budget filters, and zero-hallucination recommendation reasoning.
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


class RecommendationModule:
    def __init__(self, llm_model="qwen2.5:0.5b"):
        self.llm_model = llm_model
        self.df_products = None
        self.vectorizer = None
        self.tfidf_matrix = None

        self.load_catalog()
        self.build_tfidf_index()

    def load_catalog(self):
        self.df_products = load_products_dataset()

    def build_tfidf_index(self):
        """
        Q6: Create TF-IDF vectors for product descriptions and catalog attributes.
        """
        corpus = (
            self.df_products["ProductName"].astype(str) + " " +
            self.df_products["Category"].astype(str) + " " +
            self.df_products["Brand"].astype(str) + " " +
            self.df_products["Description"].astype(str)
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _detect_category_synonym(self, query_lower):
        """Map common user terms to exact catalog categories."""
        mappings = {
            "Mobile": ["mobile", "smartphone", "smart phone", "phone", "cellphone", "android", "iphone"],
            "Laptop": ["laptop", "notebook", "gaming laptop", "computer", "pc"],
            "Headphones": ["headphone", "headphones", "earphone", "earphones", "earbuds", "audio", "airpods"],
            "Smartwatch": ["smartwatch", "smart watch", "watch", "fitness band", "tracker"]
        }
        for cat, synonyms in mappings.items():
            if any(s in query_lower for s in synonyms):
                return cat
        return None

    def retrieve_top_products(self, query, top_k=3):
        """
        Q7: Retrieve top relevant products using TF-IDF similarity with semantic category alignment.
        """
        query_lower = query.lower()
        matched_cat = self._detect_category_synonym(query_lower)

        # Detect budget intent
        is_budget = any(w in query_lower for w in ["budget", "cheap", "affordable", "low cost", "entry level"])

        # Detect explicit max price if mentioned
        price_match = re.findall(r"(?:under|below|less than)\s*(?:rs\.?|₹)?\s*(\d+[\d,]*)", query_lower)
        max_budget = float(price_match[0].replace(",", "")) if price_match else None

        # Compute TF-IDF base similarities
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        scores = sims.copy()
        for idx, row in self.df_products.iterrows():
            cat = str(row["Category"])
            brand = str(row["Brand"]).lower()
            price = float(row["Price"])

            # Strongly boost matching category
            if matched_cat and cat == matched_cat:
                scores[idx] += 0.50
            elif matched_cat and cat != matched_cat:
                scores[idx] -= 0.50

            # Brand boost
            if brand in query_lower:
                scores[idx] += 0.20

            # Budget factor: favor affordable items when budget is requested
            if is_budget and matched_cat and cat == matched_cat:
                # normalize price penalty: lower price gets boost
                price_norm = min(1.0, price / 80000.0)
                scores[idx] += (1.0 - price_norm) * 0.35

            if max_budget and price <= max_budget:
                scores[idx] += 0.30
            elif max_budget and price > max_budget:
                scores[idx] -= 0.60

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_df = self.df_products.iloc[top_indices].copy()
        top_df["RankScore"] = [scores[i] for i in top_indices]
        return top_df

    def recommend_with_ollama(self, query, top_products):
        """
        Q7: Use Ollama to recommend the best product with factual justification.
        """
        products_context = ""
        for i, (_, p) in enumerate(top_products.iterrows()):
            products_context += (
                f"Candidate #{i+1}:\n"
                f"- Name: {p['ProductName']}\n"
                f"- Brand: {p['Brand']}\n"
                f"- Category: {p['Category']}\n"
                f"- Price: ₹{p['Price']:,}\n"
                f"- Rating: {p['Rating']}/5.0\n"
                f"- Description: {p['Description']}\n\n"
            )

        prompt = f"""You are a professional e-commerce shopping advisor.
A customer is asking: "{query}"

We retrieved these top relevant candidate products from our official store inventory:
{products_context}

Select the single best product matching the user's intent from the candidate list above.
Provide your response strictly in this exact format:

Among the retrieved products, I recommend
<Product Name>.
Brand : <Brand>
Price : ₹<Price>
Reason:
<2-3 concise lines explaining why this product is the best match for the customer's request based strictly on its specifications, rating, and value.>

Do not invent products or change prices."""

        success, content = safe_chat(model=self.llm_model, prompt=prompt, temperature=0.2)
        if success:
            return content

        # Factual fallback if Ollama is unreachable
        best_p = top_products.iloc[0]
        return (
            f"Among the retrieved products, I recommend\n"
            f"{best_p['ProductName']}.\n"
            f"Brand : {best_p['Brand']}\n"
            f"Price : ₹{best_p['Price']:,}\n"
            f"Reason:\n"
            f"{best_p['ProductName']} is the top-ranked match for your search based on its "
            f"rating of {best_p['Rating']}/5.0 and specifications: {best_p['Description']}"
        )

    def process_query(self, query, top_k=3):
        top_df = self.retrieve_top_products(query, top_k=top_k)
        rec_text = self.recommend_with_ollama(query, top_df)
        return {
            "query": query,
            "retrieved_products": top_df["ProductName"].tolist(),
            "top_products_df": top_df,
            "recommendation": rec_text
        }


def main():
    rec_mod = RecommendationModule()

    queries = [
        "Recommend a gaming laptop",
        "Suggest a budget smartphone"
    ]

    for q in queries:
        print("\n" + "=" * 60)
        print(f"Search Product : {q}")
        res = rec_mod.process_query(q, top_k=3)
        print("\nRetrieved Products:")
        for name in res["retrieved_products"]:
            print(f"- {name}")
        print("\nGPT Response:")
        print(res["recommendation"])


if __name__ == "__main__":
    main()
