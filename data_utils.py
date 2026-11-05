"""
data_utils.py
=============
Robust dataset loader, schema validator, and cleaning utility for ECommerceGPT.
Handles missing values, data type normalization, and dataset integrity checks.
"""

import os
import pandas as pd


def get_base_dir():
    """Return the base directory for locating project files."""
    return os.path.dirname(os.path.abspath(__file__))


def get_data_dir():
    """Locate the data directory whether running from root or ECommerceGPT subfolder."""
    base = get_base_dir()
    candidates = [
        os.path.join(base, "data"),
        os.path.join(base, "ECommerceGPT", "data"),
        os.path.join(base, "..", "data"),
        "data"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(base, "data"))


def load_products_dataset(data_dir=None):
    """
    Load, validate, and clean products.csv.
    Required attributes: ProductID, ProductName, Category, Brand, Price, Description, Rating.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    path = os.path.join(data_dir, "products.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Products dataset missing at '{path}'.")

    df = pd.read_csv(path)

    # Clean & normalize strings
    string_cols = ["ProductName", "Category", "Brand", "Description"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].fillna("Unknown")

    # Clean & normalize numerics
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(4.0).astype(float)
    if "ProductID" in df.columns:
        df["ProductID"] = pd.to_numeric(df["ProductID"], errors="coerce").fillna(0).astype(int)

    return df


def load_faqs_dataset(data_dir=None):
    """
    Load, validate, and clean faqs.csv.
    Required attributes: FAQID, Question, Answer.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    path = os.path.join(data_dir, "faqs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAQs dataset missing at '{path}'.")

    df = pd.read_csv(path)
    df["Question"] = df["Question"].astype(str).str.strip().fillna("")
    df["Answer"] = df["Answer"].astype(str).str.strip().fillna("")
    if "FAQID" in df.columns:
        df["FAQID"] = pd.to_numeric(df["FAQID"], errors="coerce").fillna(0).astype(int)

    return df


def load_reviews_dataset(data_dir=None):
    """
    Load, validate, and clean reviews.csv.
    Required attributes: ReviewID, ProductID, ProductName, Review, Rating.
    """
    if data_dir is None:
        data_dir = get_data_dir()
    path = os.path.join(data_dir, "reviews.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reviews dataset missing at '{path}'.")

    df = pd.read_csv(path)
    df["ProductName"] = df["ProductName"].astype(str).str.strip().fillna("Unknown")
    df["Review"] = df["Review"].astype(str).str.strip().fillna("")
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(5).astype(int)
    if "ReviewID" in df.columns:
        df["ReviewID"] = pd.to_numeric(df["ReviewID"], errors="coerce").fillna(0).astype(int)
    if "ProductID" in df.columns:
        df["ProductID"] = pd.to_numeric(df["ProductID"], errors="coerce").fillna(0).astype(int)

    return df
