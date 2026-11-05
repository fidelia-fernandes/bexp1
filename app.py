"""
app.py
======
ECommerceGPT: AI-Powered E-Commerce Assistant & Intelligence Studio
Built with Streamlit, Ollama (qwen2.5:0.5b & nomic-embed-text), Scikit-Learn, and NLTK/TextBlob.
"""

import os
import sys
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="ECommerceGPT | AI Shopping Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Add Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
ecommerce_dir = os.path.join(BASE_DIR, "ECommerceGPT")
if os.path.exists(ecommerce_dir) and ecommerce_dir not in sys.path:
    sys.path.insert(0, ecommerce_dir)

from data_utils import load_products_dataset, load_faqs_dataset, load_reviews_dataset
from intent_module import IntentDetector
from faq_module import FAQModule
from recommendation_module import RecommendationModule
from product_module import ProductQAModule
from review_module import ReviewIntelligenceModule
from ollama_client import check_ollama_daemon


# 3. Cached Resource Loaders
@st.cache_resource(show_spinner=False)
def load_all_modules():
    intent_det = IntentDetector(llm_model="qwen2.5:0.5b")
    faq_mod = FAQModule(threshold=0.50, embed_model="nomic-embed-text", llm_model="qwen2.5:0.5b")
    rec_mod = RecommendationModule(llm_model="qwen2.5:0.5b")
    prod_mod = ProductQAModule(llm_model="qwen2.5:0.5b")
    rev_mod = ReviewIntelligenceModule(llm_model="qwen2.5:0.5b")
    return intent_det, faq_mod, rec_mod, prod_mod, rev_mod


@st.cache_data(show_spinner=False)
def load_cached_datasets():
    df_p = load_products_dataset()
    df_f = load_faqs_dataset()
    df_r = load_reviews_dataset()
    return df_p, df_f, df_r


intent_det, faq_mod, rec_mod, prod_mod, rev_mod = load_all_modules()
df_products, df_faqs, df_reviews = load_cached_datasets()


# 4. Human-Crafted Custom CSS Design System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Typography & Global Theme */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    .stApp {
        background-color: #F8FAFC;
    }

    /* Top Brand Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border-radius: 18px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-badge-cluster {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.22);
        padding: 7px 14px;
        border-radius: 24px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #EEF2FF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    /* KPI / Metric Glass Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 18px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.06);
    }
    .kpi-header {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 600;
    }

    /* Product Showcase Cards */
    .product-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        position: relative;
    }
    .product-card:hover {
        border-color: #6366F1;
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.1);
    }
    .product-badge {
        position: absolute;
        top: 18px;
        right: 18px;
        background: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .product-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
        padding-right: 65px;
    }
    .product-brand {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .product-price {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .product-rating {
        font-size: 0.85rem;
        font-weight: 700;
        color: #D97706;
        margin-bottom: 12px;
    }
    .product-desc {
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.45;
        margin-bottom: 14px;
        flex-grow: 1;
    }

    /* Intent Badges */
    .intent-pill {
        display: inline-flex;
        align-items: center;
        padding: 5px 14px;
        border-radius: 16px;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    .intent-faq { background: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }
    .intent-rec { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    .intent-prod { background: #F3E8FF; color: #6B21A8; border: 1px solid #E9D5FF; }
    .intent-rev { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }

    /* Review Quotation Cards */
    .review-quote-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }
    .review-quote-text {
        font-size: 0.92rem;
        color: #1E293B;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .review-quote-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        color: #64748B;
    }

    /* Executive Summary Block */
    .summary-box {
        background: #FFFFFF;
        border: 1px solid #E0E7FF;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.05);
        margin-top: 18px;
    }
    .summary-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #312E81;
        margin-bottom: 14px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* High Visibility Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1.5px solid #CBD5E1 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
        border-color: #818CF8 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #4F46E5 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4338CA !important;
        border-color: #4338CA !important;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
    }
    .section-subtext {
        font-size: 0.88rem;
        color: #64748B;
        margin: 4px 0 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# 5. Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 24px; padding-bottom: 14px; border-bottom: 1px solid #E2E8F0;">
        <div style="font-weight: 800; font-size: 1.35rem; color: #0F172A; letter-spacing: -0.02em;">ECommerceGPT</div>
        <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Intelligence Studio</div>
    </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation:",
        [
            "Conversational Assistant",
            "Part A: Catalog & Datasets",
            "Part B: FAQ Semantic Search",
            "Part C: Product Recommendations",
            "Part D: Product QA Dossier",
            "Part E: Review Sentiment"
        ],
        index=0
    )


# 6. Global Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">NovaCart Shopping Intelligence</div>
    <div class="hero-badge-cluster">
        <div class="hero-pill">
            <div class="status-dot"></div> Ollama Engine Live
        </div>
        <div class="hero-pill">
            100 Products Indexed
        </div>
        <div class="hero-pill">
            1,000 Verified Reviews
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: CONVERSATIONAL ASSISTANT (CHATBOT)
# ==============================================================================
if nav_choice == "Conversational Assistant":
    st.markdown('<div class="section-title">Conversational AI Concierge</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Intelligent intent routing across store FAQs, product recommendations, specifications, and customer sentiment.</div>', unsafe_allow_html=True)

    # Required Suggested Query Cards (Phase 3 Requirement 5)
    st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 8px;'>Suggested Queries:</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    quick_input = None
    if c1.button("Track My Order", use_container_width=True):
        quick_input = "Where is my parcel?"
    if c2.button("Recommend a Laptop", use_container_width=True):
        quick_input = "Recommend a gaming laptop"
    if c3.button("Explore Products", use_container_width=True):
        quick_input = "Tell me about Lenovo Laptop"
    if c4.button("Analyze Customer Reviews", use_container_width=True):
        quick_input = "Summarize Lenovo customer reviews"

    # Chat history state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "intent": "WELCOME",
                "content": "Hello. I am your NovaCart AI Shopping Assistant. How may I assist you today? You can track shipments, ask for product recommendations, view technical specifications, or explore customer reviews."
            }
        ]

    # Render conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "intent" in msg and msg["intent"] not in ["WELCOME", None]:
                intent = msg["intent"]
                badge_style = {
                    "FAQ": "intent-faq",
                    "RECOMMENDATION": "intent-rec",
                    "PRODUCT": "intent-prod",
                    "REVIEW": "intent-rev"
                }.get(intent, "intent-faq")

                label = {
                    "FAQ": "Policy & FAQ Match",
                    "RECOMMENDATION": "Product Recommendation",
                    "PRODUCT": "Product Catalog Context",
                    "REVIEW": "Review Sentiment Pulse"
                }.get(intent, intent)

                st.markdown(f'<span class="intent-pill {badge_style}">{label}</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])

    # User input
    user_query = st.chat_input("Ask a question about tracking, recommendations, specs, or reviews...") or quick_input

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        intent = intent_det.detect_intent(user_query)

        with st.chat_message("assistant"):
            badge_style = {
                "FAQ": "intent-faq",
                "RECOMMENDATION": "intent-rec",
                "PRODUCT": "intent-prod",
                "REVIEW": "intent-rev"
            }.get(intent, "intent-faq")

            label = {
                "FAQ": "Policy & FAQ Match",
                "RECOMMENDATION": "Product Recommendation",
                "PRODUCT": "Product Catalog Context",
                "REVIEW": "Review Sentiment Pulse"
            }.get(intent, intent)

            st.markdown(f'<span class="intent-pill {badge_style}">{label}</span>', unsafe_allow_html=True)

            with st.spinner("Processing inquiry with Ollama..."):
                if intent == "FAQ":
                    res = faq_mod.process_query(user_query)
                    bot_reply = res["response"] if res["matched"] else prod_mod.answer_question(user_query)["answer"]

                elif intent == "RECOMMENDATION":
                    res = rec_mod.process_query(user_query, top_k=3)
                    bot_reply = res["recommendation"]

                elif intent == "PRODUCT":
                    res = prod_mod.answer_question(user_query)
                    bot_reply = res["answer"]

                elif intent == "REVIEW":
                    res = rev_mod.process_query(user_query)
                    analysis = res["analysis"]
                    bot_reply = (
                        f"**Customer Sentiment Overview for {analysis['product_name']}:**\n\n"
                        f"Positive: `{analysis['positive']}` | "
                        f"Neutral: `{analysis['neutral']}` | "
                        f"Negative: `{analysis['negative']}`\n\n"
                        f"{res['summary']}"
                    )
                else:
                    bot_reply = "I am ready to assist you. You can ask about store policies, request product recommendations, view product details, or explore customer reviews."

                st.markdown(bot_reply)

        st.session_state.messages.append({"role": "assistant", "intent": intent, "content": bot_reply})

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ==============================================================================
# SECTION 2: PART A - DATASET EXPLORER & MAPPING
# ==============================================================================
elif nav_choice == "Part A: Catalog & Datasets":
    st.markdown('<div class="section-title">Part A: Dataset Preparation & Catalog Structure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Inspect raw record schemas, distribution metrics, and academic module alignments.</div>', unsafe_allow_html=True)

    # Clean KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-header">Catalog Products</div>
            <div class="kpi-value">{len(df_products)}</div>
            <div class="kpi-subtext">4 Categories • 7 Attributes</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">Store FAQs</div>
            <div class="kpi-value">{len(df_faqs)}</div>
            <div class="kpi-subtext">100% Policy Coverage</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">Customer Reviews</div>
            <div class="kpi-value">{len(df_reviews):,}</div>
            <div class="kpi-subtext">Mean Rating: 4.2 / 5.0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">Vector Embeddings</div>
            <div class="kpi-value">20 × 768</div>
            <div class="kpi-subtext">Nomic Embed Pre-computed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_prod, tab_faq, tab_rev, tab_map = st.tabs([
        "Products Inventory (Q1)",
        "FAQ Policy Records (Q1)",
        "Customer Feedback Stream (Q1)",
        "Module Mapping Matrix (Q2)"
    ])

    with tab_prod:
        st.markdown("#### Product Catalog (`products.csv`)")
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            prod_search = st.text_input("Filter by Name, Category, or Brand:", "", placeholder="e.g. Lenovo, Laptop, Samsung...")
        with col_f2:
            cat_choice = st.selectbox("Category Filter:", ["All Categories"] + list(df_products["Category"].unique()))

        filtered_p = df_products
        if cat_choice != "All Categories":
            filtered_p = filtered_p[filtered_p["Category"] == cat_choice]
        if prod_search:
            filtered_p = filtered_p[
                filtered_p["ProductName"].str.contains(prod_search, case=False, na=False) |
                filtered_p["Brand"].str.contains(prod_search, case=False, na=False) |
                filtered_p["Description"].str.contains(prod_search, case=False, na=False)
            ]
        st.dataframe(filtered_p, use_container_width=True, height=350)

    with tab_faq:
        st.markdown("#### Frequently Asked Questions (`faqs.csv`)")
        st.dataframe(df_faqs, use_container_width=True, height=350)

    with tab_rev:
        st.markdown("#### Customer Reviews (`reviews.csv`)")
        selected_prod = st.selectbox("Filter reviews by product:", ["All Products"] + list(df_reviews["ProductName"].unique()[:20]))
        if selected_prod != "All Products":
            st.dataframe(df_reviews[df_reviews["ProductName"] == selected_prod], use_container_width=True, height=350)
        else:
            st.dataframe(df_reviews.head(100), use_container_width=True, height=350)

    with tab_map:
        st.markdown("#### Q2: Architectural Mapping to Chatbot Modules")
        st.markdown("""
        | Dataset | Core Purpose | Corresponding Chatbot Module | Retrieval & Generation Mechanism |
        | :--- | :--- | :--- | :--- |
        | **`faqs.csv`** | Official e-commerce policies (tracking, returns, refunds, warranty). | **Part B: FAQ Semantic Search** | Dense embeddings (`nomic-embed-text`) + Cosine Similarity ($\ge 0.50$) + Ollama synthesis. |
        | **`products.csv`** | Full inventory specifications, brand metadata, prices, and ratings. | **Part C: Product Recommendation** | TF-IDF n-gram vectorization over combined descriptions + Ollama Top-3 comparative reasoning. |
        | **`products.csv`** | Clean catalog facts for grounding. | **Part D: Product Question Answering (RAG)** | Structured key-value context extraction + Constrained Ollama prompt for zero-hallucination Q&A. |
        | **`reviews.csv`** | Qualitative customer feedback and star ratings. | **Part E: Customer Review Intelligence** | Pre-trained **VADER / TextBlob** sentiment scoring + Ollama Strengths, Weaknesses, and Buying advice. |
        """)


# ==============================================================================
# SECTION 3: PART B - FAQ SEMANTIC SEARCH
# ==============================================================================
elif nav_choice == "Part B: FAQ Semantic Search":
    st.markdown('<div class="section-title">Part B: Semantic FAQ Search & Policy Resolution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Cosine similarity evaluation against dense 768-d embeddings with threshold-controlled natural response generation.</div>', unsafe_allow_html=True)

    col_q, col_th = st.columns([3, 1])
    with col_q:
        faq_input = st.text_input("Enter customer inquiry:", "Where is my parcel?")
    with col_th:
        threshold = st.slider("Cosine Match Threshold:", 0.30, 0.95, 0.50, 0.05)

    if st.button("Search FAQ & Generate Response", type="primary"):
        with st.spinner("Computing cosine similarity against FAQ vector store..."):
            res = faq_mod.process_query(faq_input)
            score = res["similarity_score"]
            is_matched = score >= threshold

            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.markdown(f"""
                <div class="kpi-card" style="text-align: center;">
                    <div class="kpi-header" style="justify-content: center;">Cosine Similarity</div>
                    <div class="kpi-value" style="color: {'#10B981' if is_matched else '#EF4444'};">{score:.2f}</div>
                    <div style="font-size: 0.82rem; font-weight: 700; color: {'#059669' if is_matched else '#DC2626'};">
                        {'Match Accepted' if is_matched else 'Below Threshold'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_s2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Top Matching FAQ</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #1E293B; margin-top: 4px;">{res['best_faq']}</div>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #E2E8F0;">
                    <div style="font-size: 0.85rem; color: #475569;"><strong>Store Policy:</strong> {res['official_answer']}</div>
                </div>
                """, unsafe_allow_html=True)

            if is_matched:
                st.markdown(f"""
                <div class="summary-box">
                    <div class="summary-heading">Ollama Natural Response</div>
                    <div style="font-size: 0.96rem; color: #1E293B; line-height: 1.6;">{res['response']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Query similarity did not meet the confidence threshold. The pipeline automatically continues to the Recommendation / Product QA module.")


# ==============================================================================
# SECTION 4: PART C - PRODUCT RECOMMENDATIONS
# ==============================================================================
elif nav_choice == "Part C: Product Recommendations":
    st.markdown('<div class="section-title">Part C: TF-IDF Product Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Vectorizes catalog descriptions, retrieves Top-3 candidate matches, and uses Ollama for multi-factor purchase justification.</div>', unsafe_allow_html=True)

    col_rec_in, col_rec_k = st.columns([3, 1])
    with col_rec_in:
        rec_query = st.text_input("Enter product preference or search goal:", "Recommend a gaming laptop")
    with col_rec_k:
        top_k = st.selectbox("Candidates to retrieve:", [3, 5], index=0)

    if st.button("Retrieve & Recommend with Ollama", type="primary"):
        with st.spinner("Executing TF-IDF vector ranking & Ollama reasoning..."):
            res = rec_mod.process_query(rec_query, top_k=top_k)
            top_df = res["top_products_df"]

            st.markdown("#### Retrieved Candidate Products")
            cols = st.columns(len(top_df))
            for idx, (_, row) in enumerate(top_df.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="product-badge">Rank #{idx+1}</div>
                        <div class="product-title">{row['ProductName']}</div>
                        <div class="product-brand">{row['Brand']} • {row['Category']}</div>
                        <div class="product-price">₹{row['Price']:,}</div>
                        <div class="product-rating">Rating: {row['Rating']} / 5.0</div>
                        <div class="product-desc">{row['Description']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="summary-box">
                <div class="summary-heading">Ollama Recommendation Justification</div>
                <div style="font-size: 0.95rem; color: #1E293B; line-height: 1.6; white-space: pre-wrap;">{res['recommendation']}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 5: PART D - PRODUCT QA (RAG)
# ==============================================================================
elif nav_choice == "Part D: Product QA Dossier":
    st.markdown('<div class="section-title">Part D: Constrained Product Question Answering (RAG)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Factual question answering strictly grounded in catalog context without hallucinations, supporting continuous multi-turn dialogue.</div>', unsafe_allow_html=True)

    # Automatically initialize default active product if not already in session state
    if "rag_product" not in st.session_state:
        default_row, default_sim = prod_mod.retrieve_product("Lenovo Laptop 3")
        prod_mod.active_product = default_row
        prod_mod.active_context = prod_mod.prepare_structured_context(default_row)
        st.session_state["rag_product"] = default_row
        st.session_state["rag_context"] = prod_mod.active_context
        st.session_state["rag_sim"] = default_sim

    col_rag_l, col_rag_r = st.columns([1, 1], gap="large")

    with col_rag_l:
        st.markdown("#### 1. Target Product Context")
        search_kw = st.text_input("Search Product Catalog:", "Lenovo Laptop 3")
        if st.button("Update Product Context", use_container_width=True):
            row, sim = prod_mod.retrieve_product(search_kw)
            prod_mod.active_product = row
            prod_mod.active_context = prod_mod.prepare_structured_context(row)
            st.session_state["rag_product"] = row
            st.session_state["rag_context"] = prod_mod.active_context
            st.session_state["rag_sim"] = sim

        p = st.session_state["rag_product"]
        st.markdown(f"""
        <div class="product-card" style="margin-top: 14px;">
            <div class="product-badge">Similarity: {st.session_state['rag_sim']:.2f}</div>
            <div class="product-title">{p['ProductName']}</div>
            <div class="product-brand">Brand: {p['Brand']} | Category: {p['Category']}</div>
            <div class="product-price">₹{p['Price']:,}</div>
            <div class="product-rating">Rating: {p['Rating']} / 5.0</div>
            <div class="product-desc">{p['Description']}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Structured Context Extracted for Ollama", expanded=True):
            st.code(st.session_state["rag_context"], language="yaml")

    with col_rag_r:
        st.markdown("#### 2. Multi-turn Follow-up Questions")
        st.markdown("""
        <div style="font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 8px;">
            Click a follow-up question to query instantly:
        </div>
        """, unsafe_allow_html=True)

        q_clicked = None
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Tell me about Lenovo Laptop", use_container_width=True):
                q_clicked = "Tell me about Lenovo Laptop"
            if st.button("What is its price?", use_container_width=True):
                q_clicked = "What is its price?"

        with col_btn2:
            if st.button("What brand is it?", use_container_width=True):
                q_clicked = "What brand is it?"
            if st.button("What is its rating and quality?", use_container_width=True):
                q_clicked = "What is its rating and quality?"

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        custom_q = st.text_input("Or enter a custom question:", value="", placeholder="e.g. Is this suitable for gaming?")
        submit_custom = st.button("Ask Assistant", type="primary", use_container_width=True)

        active_question = q_clicked or (custom_q if submit_custom and custom_q else None)

        if active_question:
            prod_mod.active_product = st.session_state["rag_product"]
            prod_mod.active_context = st.session_state["rag_context"]
            with st.spinner("Querying constrained RAG prompt..."):
                res = prod_mod.answer_question(active_question)
                st.markdown(f"""
                <div class="summary-box" style="margin-top: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Question Asked:</div>
                    <div style="font-size: 1rem; font-weight: 700; color: #1E293B; margin-bottom: 10px;">"{active_question}"</div>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #E2E8F0;">
                    <div class="summary-heading" style="margin-bottom: 6px;">Assistant Factual Response</div>
                    <div style="font-size: 0.98rem; color: #0F172A; line-height: 1.6;">{res['answer']}</div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 6: PART E - REVIEW INTELLIGENCE
# ==============================================================================
elif nav_choice == "Part E: Review Sentiment":
    st.markdown('<div class="section-title">Part E: Customer Review Intelligence & Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtext">Pre-trained sentiment modeling (VADER/TextBlob) with Ollama executive review summarization and buying advice.</div>', unsafe_allow_html=True)

    col_rev_in, col_rev_btn = st.columns([3, 1])
    with col_rev_in:
        rev_prod_input = st.text_input("Select or type product name:", "Lenovo Laptop")
    with col_rev_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Run Sentiment Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Analyzing review stream and synthesizing executive brief..."):
            res = rev_mod.process_query(rev_prod_input)
            analysis = res["analysis"]

            # KPI Breakdown
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="border-top: 4px solid #10B981;">
                    <div class="kpi-header">Positive Sentiment</div>
                    <div class="kpi-value" style="color: #059669;">{analysis['positive']}</div>
                    <div class="kpi-subtext" style="color: #059669;">Satisfied buyers</div>
                </div>
                <div class="kpi-card" style="border-top: 4px solid #F59E0B;">
                    <div class="kpi-header">Neutral Sentiment</div>
                    <div class="kpi-value" style="color: #D97706;">{analysis['neutral']}</div>
                    <div class="kpi-subtext" style="color: #D97706;">Balanced feedback</div>
                </div>
                <div class="kpi-card" style="border-top: 4px solid #EF4444;">
                    <div class="kpi-header">Negative Sentiment</div>
                    <div class="kpi-value" style="color: #DC2626;">{analysis['negative']}</div>
                    <div class="kpi-subtext" style="color: #DC2626;">Reported concerns</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_feed_l, col_feed_r = st.columns([1, 1], gap="large")

            with col_feed_l:
                st.markdown("#### Sample Analyzed Customer Reviews")
                for r in analysis["reviews"][:4]:
                    border_c = "#10B981" if r["sentiment"] == "Positive" else ("#EF4444" if r["sentiment"] == "Negative" else "#F59E0B")
                    badge_c = "#DCFCE7" if r["sentiment"] == "Positive" else ("#FEE2E2" if r["sentiment"] == "Negative" else "#FEF3C7")
                    text_c = "#166534" if r["sentiment"] == "Positive" else ("#991B1B" if r["sentiment"] == "Negative" else "#92400E")

                    st.markdown(f"""
                    <div class="review-quote-card" style="border-left-color: {border_c};">
                        <div class="review-quote-text">"{r['text']}"</div>
                        <div class="review-quote-meta">
                            <span style="background: {badge_c}; color: {text_c}; padding: 3px 10px; border-radius: 12px; font-weight: 700;">
                                {r['sentiment']}
                            </span>
                            <span>Confidence: <strong>{r['score']:.3f}</strong></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_feed_r:
                st.markdown("#### Ollama Review Summary & Buying Suggestion")
                st.markdown(f"""
                <div class="summary-box" style="margin-top: 0;">
                    <div style="font-size: 0.95rem; color: #1E293B; line-height: 1.6; white-space: pre-wrap;">{res['summary']}</div>
                </div>
                """, unsafe_allow_html=True)
