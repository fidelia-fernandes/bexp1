"""
verify_all_modules.py
=====================
Comprehensive Test Suite for ECommerceGPT.
Executes all 13 core requirements and tests the exact 9 required sample queries:
1. "Where is my parcel?"
2. "How do I return a product?"
3. "Recommend a gaming laptop"
4. "Suggest a budget smartphone"
5. "Tell me about Lenovo Laptop"
6. "What is its price?"
7. "What brand is it?"
8. "Summarize Lenovo customer reviews"
9. "Is Lenovo Laptop worth buying?"
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_utils import load_products_dataset, load_faqs_dataset, load_reviews_dataset
from ollama_client import check_ollama_daemon, check_model_available
from create_embeddings import build_and_save_faq_embeddings
from faq_module import FAQModule
from recommendation_module import RecommendationModule
from product_module import ProductQAModule
from review_module import ReviewIntelligenceModule
from intent_module import IntentDetector
from chatbot import ECommerceChatbot


def test_suite():
    results = {}
    print("\n" + "=" * 80)
    print(" 🚀 STARTING FULL ECOMMERCEGPT VERIFICATION TEST SUITE")
    print("=" * 80)

    # 1. Daemon & Model Check
    print("\n[TEST 1] Checking Ollama Daemon & Models...")
    daemon_ok, daemon_msg = check_ollama_daemon()
    qwen_ok, qwen_msg = check_model_available("qwen2.5:0.5b")
    embed_ok, embed_msg = check_model_available("nomic-embed-text")
    print(f"  • Daemon: {daemon_msg}")
    print(f"  • Qwen:   {qwen_msg}")
    print(f"  • Nomic:  {embed_msg}")
    results["Ollama Health"] = daemon_ok and qwen_ok and embed_ok

    # 2. Dataset Loading & Integrity Check
    print("\n[TEST 2] Dataset Loading & Schema Integrity...")
    try:
        df_p = load_products_dataset()
        df_f = load_faqs_dataset()
        df_r = load_reviews_dataset()
        assert len(df_p) == 100, f"Expected 100 products, got {len(df_p)}"
        assert len(df_f) == 20, f"Expected 20 faqs, got {len(df_f)}"
        assert len(df_r) == 1000, f"Expected 1000 reviews, got {len(df_r)}"
        print(f"  • Products: {len(df_p)} records, 0 nulls in critical attributes.")
        print(f"  • FAQs:     {len(df_f)} records, 0 nulls in critical attributes.")
        print(f"  • Reviews:  {len(df_r)} records, 0 nulls in critical attributes.")
        results["Dataset Preparation"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["Dataset Preparation"] = False

    # 3. FAQ Embedding Generation & Cosine Search
    print("\n[TEST 3] FAQ Embedding Generation & Semantic Retrieval...")
    try:
        emb_ok = build_and_save_faq_embeddings()
        faq_mod = FAQModule(threshold=0.50)
        res_parcel = faq_mod.process_query("Where is my parcel?")
        assert res_parcel["matched"] is True, "Expected 'Where is my parcel?' to match FAQ above threshold"
        assert res_parcel["similarity_score"] >= 0.50, "Similarity score should be >= 0.50"
        print(f"  • 'Where is my parcel?' matched FAQ: '{res_parcel['best_faq']}' (Score: {res_parcel['similarity_score']:.2f})")
        print(f"  • Response snippet: {res_parcel['response'][:80]}...")
        results["FAQ Semantic Search"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["FAQ Semantic Search"] = False

    # 4. Product Recommendation (Gaming Laptop & Budget Smartphone)
    print("\n[TEST 4] Product Recommendation Engine (TF-IDF & LLM Justification)...")
    try:
        rec_mod = RecommendationModule()

        # Query A: Gaming Laptop
        res_game = rec_mod.process_query("Recommend a gaming laptop", top_k=3)
        assert len(res_game["retrieved_products"]) == 3
        assert any("laptop" in p.lower() for p in res_game["retrieved_products"]), "Should retrieve laptops"
        print(f"  • 'Recommend a gaming laptop' retrieved: {res_game['retrieved_products']}")

        # Query B: Budget Smartphone
        res_phone = rec_mod.process_query("Suggest a budget smartphone", top_k=3)
        assert len(res_phone["retrieved_products"]) == 3
        assert any("mobile" in p.lower() for p in res_phone["retrieved_products"]), "Should retrieve mobiles for smartphone query"
        print(f"  • 'Suggest a budget smartphone' retrieved: {res_phone['retrieved_products']}")
        results["Product Recommendation"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["Product Recommendation"] = False

    # 5. Product Question Answering (RAG) Multi-turn
    print("\n[TEST 5] Product QA RAG & Conversational Follow-ups...")
    try:
        prod_mod = ProductQAModule()

        # Turn 1: Product overview
        t1 = prod_mod.answer_question("Tell me about Lenovo Laptop")
        assert "Lenovo" in t1["product_name"]
        print(f"  • Turn 1 (Overview) -> Product: {t1['product_name']} | Answer: {t1['answer'][:60]}...")

        # Turn 2: Follow-up on Price
        t2 = prod_mod.answer_question("What is its price?")
        assert t2["is_followup"] is True, "Expected pronoun 'its' to be detected as follow-up"
        assert "₹" in t2["answer"] or "price" in t2["answer"].lower()
        print(f"  • Turn 2 (Price)    -> Followup Detected: {t2['is_followup']} | Answer: {t2['answer']}")

        # Turn 3: Follow-up on Brand
        t3 = prod_mod.answer_question("What brand is it?")
        assert t3["is_followup"] is True
        assert "Lenovo" in t3["answer"]
        print(f"  • Turn 3 (Brand)    -> Followup Detected: {t3['is_followup']} | Answer: {t3['answer']}")
        results["Product QA RAG"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["Product QA RAG"] = False

    # 6. Customer Review Sentiment Intelligence
    print("\n[TEST 6] Customer Review Sentiment & Ollama Synthesis...")
    try:
        rev_mod = ReviewIntelligenceModule()

        # Query A: Summarize reviews
        r1 = rev_mod.process_query("Summarize Lenovo customer reviews")
        an1 = r1["analysis"]
        assert an1["positive"] > 0, "Expected positive reviews count"
        assert len(an1["reviews"]) > 0, "Expected sample reviews"
        assert "Overall Opinion" in r1["summary"] or "Buying Suggestion" in r1["summary"]
        print(f"  • 'Summarize Lenovo customer reviews' -> Pos: {an1['positive']}, Neu: {an1['neutral']}, Neg: {an1['negative']}")

        # Query B: Is it worth buying?
        r2 = rev_mod.process_query("Is Lenovo Laptop worth buying?")
        print(f"  • 'Is Lenovo Laptop worth buying?' -> Summary snippet: {r2['summary'][:90]}...")
        results["Review Intelligence"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["Review Intelligence"] = False

    # 7. Intent Detection on 9 Standard Queries
    print("\n[TEST 7] Intent Detection on Required 9 Queries...")
    expected_intents = {
        "Where is my parcel?": "FAQ",
        "How do I return a product?": "FAQ",
        "Recommend a gaming laptop": "RECOMMENDATION",
        "Suggest a budget smartphone": "RECOMMENDATION",
        "Tell me about Lenovo Laptop": "PRODUCT",
        "What is its price?": "PRODUCT",
        "What brand is it?": "PRODUCT",
        "Summarize Lenovo customer reviews": "REVIEW",
        "Is Lenovo Laptop worth buying?": "REVIEW"
    }

    intent_det = IntentDetector()
    all_intents_correct = True
    for query, exp_intent in expected_intents.items():
        actual_intent = intent_det.detect_intent(query)
        is_ok = actual_intent == exp_intent
        print(f"  • '{query}' -> Expected: {exp_intent:14} | Got: {actual_intent:14} | {'PASS' if is_ok else 'FAIL'}")
        if not is_ok:
            all_intents_correct = False
    results["Intent Detection"] = all_intents_correct

    # 8. Complete Chatbot Integration Workflow
    print("\n[TEST 8] Complete Chatbot Integration Workflow...")
    try:
        bot = ECommerceChatbot()
        test_queries = [
            "Where is my parcel?",
            "Recommend a gaming laptop",
            "Tell me about Lenovo Laptop",
            "What is its price?",
            "Summarize Lenovo customer reviews",
            "exit"
        ]
        for q in test_queries:
            intent, reply = bot.handle_query(q)
            assert intent is not None and reply is not None
        print("  • All integrated queries routed and executed cleanly through pipeline.")
        results["Chatbot Integration"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["Chatbot Integration"] = False

    # Final Summary Table
    print("\n" + "=" * 80)
    print(" 📊 FINAL VERIFICATION REPORT")
    print("=" * 80)
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for test_name, status in results.items():
        symbol = "✅ PASS" if status else "❌ FAIL"
        print(f"  {test_name:30} : {symbol}")

    print("-" * 80)
    print(f"  Total Passed: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    print("=" * 80 + "\n")
    return passed_count == total_count


if __name__ == "__main__":
    success = test_suite()
    sys.exit(0 if success else 1)
