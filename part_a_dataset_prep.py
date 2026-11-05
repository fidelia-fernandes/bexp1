"""
part_a_dataset_prep.py
======================
Part A - Dataset Preparation:
Q1. Load Products, FAQs and Reviews datasets. Handle missing values, clean data,
    and display first five records, rows, columns and attributes.
Q2. Explain the purpose of each dataset and map it to the chatbot modules.
"""

import os
import sys
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_utils import load_products_dataset, load_faqs_dataset, load_reviews_dataset


def inspect_dataframe(df, dataset_name):
    print("=" * 80)
    print(f" DATASET: {dataset_name}")
    print("=" * 80)
    print(f"Total Rows (Records) : {df.shape[0]}")
    print(f"Total Columns        : {df.shape[1]}")
    print(f"Attributes (Columns) : {df.columns.tolist()}\n")

    print("Data Types & Null Value Validation:")
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        print(f"  • {col:15} | Type: {str(df[col].dtype):10} | Nulls: {null_count}")

    print("\nFirst 5 Records:")
    print("-" * 80)
    print(df.head(5).to_string(index=False))
    print("-" * 80)
    print()


def main():
    print("\n" + "=" * 80)
    print(" EXPERIMENT 1: PART A - DATASET PREPARATION & EXPLORATION")
    print("=" * 80 + "\n")

    # Q1: Load, Clean, & Display
    df_products = load_products_dataset()
    inspect_dataframe(df_products, "Products Catalog (products.csv)")

    df_faqs = load_faqs_dataset()
    inspect_dataframe(df_faqs, "Frequently Asked Questions (faqs.csv)")

    df_reviews = load_reviews_dataset()
    inspect_dataframe(df_reviews, "Customer Reviews (reviews.csv)")

    # Q2: Purpose & Mapping Explanation
    print("=" * 80)
    print(" Q2. DATASET PURPOSE AND CHATBOT MODULE MAPPING")
    print("=" * 80)

    explanation = """
1. Products Dataset (products.csv):
   • Purpose: Contains inventory catalog data with specifications, category, brand,
     pricing, ratings, and descriptions.
   • Chatbot Module Mapping:
     - Recommendation Module (Part C): Used for TF-IDF vectorization, similarity retrieval,
       and Ollama product recommendations with factual justification.
     - Product QA / RAG Module (Part D): Supplies structured metadata context (Brand, Price,
       Specs, Rating) used by Ollama to answer product-specific questions without hallucinations.

2. FAQs Dataset (faqs.csv):
   • Purpose: Stores official store policies and answers (tracking, shipping, returns,
     refunds, warranty, payment methods).
   • Chatbot Module Mapping:
     - FAQ Semantic Search Module (Part B): Dense embeddings (nomic-embed-text) and TF-IDF
       are matched via cosine similarity, and Ollama synthesizes natural customer-facing policy answers.

3. Customer Reviews Dataset (reviews.csv):
   • Purpose: Captures qualitative feedback, user opinions, and product star ratings.
   • Chatbot Module Mapping:
     - Customer Review Intelligence Module (Part E): Pre-trained sentiment analysis models
       (VADER / TextBlob) evaluate each review for sentiment polarity and confidence. Ollama
       then synthesizes the review stream into structured Overall Opinions, Strengths, Weaknesses,
       and Buying Suggestions.
"""
    print(explanation)
    print("=" * 80)


if __name__ == "__main__":
    main()
