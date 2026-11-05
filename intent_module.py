"""
intent_module.py
================
Part F - Q12: Intelligent Intent Detection using Ollama.
Classifies user queries into exactly four categories:
1. FAQ
2. RECOMMENDATION
3. PRODUCT
4. REVIEW
Plus session termination handling (EXIT).
Uses structured few-shot prompting with Ollama and robust pattern validation.
"""

import os
import sys
import re

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add module paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ollama_client import safe_chat, check_ollama_daemon


class IntentDetector:
    def __init__(self, llm_model="qwen2.5:0.5b"):
        self.llm_model = llm_model
        self.valid_intents = ["FAQ", "RECOMMENDATION", "PRODUCT", "REVIEW"]

    def _rule_match(self, q_lower):
        """High-precision pattern matcher for common e-commerce conversational patterns."""
        # 1. Exit Check
        if q_lower in ["exit", "quit", "bye", "stop", "close", "end", "terminate"]:
            return "EXIT"

        # 2. Review Signals (highest priority for opinion/feedback)
        review_patterns = [
            r"\breviews?\b", r"\bratings?\b", r"\bcustomer\b", r"\bfeedback\b",
            r"\bsentiment\b", r"\bworth buying\b", r"\bpros and cons\b",
            r"\bopinion\b", r"\bis it good\b", r"\bshould i buy\b", r"\bsummarize.*reviews?\b"
        ]
        if any(re.search(p, q_lower) for p in review_patterns):
            return "REVIEW"

        # 3. Recommendation Signals
        rec_patterns = [
            r"\brecommend\b", r"\brecommendation\b", r"\bsuggest\b", r"\bsuggestion\b",
            r"\bbest\b", r"\btop\s*\d+", r"\bbudget\b", r"\bcheapest?\b", r"\baffordable\b",
            r"\bunder\s*(?:rs|₹|\d)", r"\bwhich\s+(?:laptop|phone|mobile|product|headphone)\b"
        ]
        if any(re.search(p, q_lower) for p in rec_patterns):
            return "RECOMMENDATION"

        # 4. FAQ Policy Signals
        faq_patterns = [
            r"\bparcel\b", r"\btrack(?:ing)?\b", r"\bshipment\b", r"\bdelivery\b",
            r"\border\s*(?:status|id)?\b", r"\breturn\b", r"\brefund\b", r"\bcancel\b",
            r"\bpayment\b", r"\bcod\b", r"\bcash on delivery\b", r"\bwarranty\b",
            r"\binvoice\b", r"\bcoupon\b", r"\bexchange\b", r"\bsupport\b",
            r"\bcontact\b", r"\bhow long\b", r"\bship internationally\b",
            r"\bchange address\b", r"\bpassword\b", r"\bpreorder\b"
        ]
        if any(re.search(p, q_lower) for p in faq_patterns):
            return "FAQ"

        # 5. Product Specs, Price, and Follow-up Signals
        product_patterns = [
            r"\bprice\b", r"\bcost\b", r"\bhow much\b", r"\bbrand\b", r"\bwho makes\b",
            r"\bspecs?\b", r"\bspecifications?\b", r"\btell me about\b", r"\bfeatures?\b",
            r"\bwhat is its\b", r"\bwhat is the\b", r"\bdetails\b",
            r"\blenovo\b", r"\bsamsung\b", r"\bapple\b", r"\bdell\b", r"\bhp\b",
            r"\basus\b", r"\bboat\b", r"\bsony\b", r"\bnoise\b", r"\bxiaomi\b", r"\brealme\b"
        ]
        if any(re.search(p, q_lower) for p in product_patterns):
            return "PRODUCT"

        return None

    def detect_intent(self, user_query):
        """
        Q12: Classify user query using Ollama with structured few-shot prompting and validation.
        """
        q_clean = user_query.strip().lower()
        if not q_clean:
            return "FAQ"

        # Fast exact check
        fast_result = self._rule_match(q_clean)
        if fast_result is not None:
            return fast_result

        # Ollama LLM Classifier with Structured Prompt
        prompt = f"""You are an intent classification engine for an e-commerce platform.
Classify the user's message into exactly ONE category from: [FAQ, RECOMMENDATION, PRODUCT, REVIEW].

Rules:
- FAQ: Inquiries about order tracking, parcel location, delivery, returns, cancellations, refunds, warranty, payment options, store policies.
- RECOMMENDATION: Inquiries asking for product suggestions, best options, budget filtering, or top items.
- PRODUCT: Inquiries about a specific product name, brand, price, features, specifications, or follow-up details.
- REVIEW: Inquiries about customer opinions, reviews, feedback, satisfaction, ratings, or pros and cons.

Examples:
"Where is my parcel?" -> FAQ
"How do I return a product?" -> FAQ
"Recommend a gaming laptop" -> RECOMMENDATION
"Suggest a budget smartphone" -> RECOMMENDATION
"Tell me about Lenovo Laptop" -> PRODUCT
"What is its price?" -> PRODUCT
"What brand is it?" -> PRODUCT
"Summarize Lenovo customer reviews" -> REVIEW
"Is Lenovo Laptop worth buying?" -> REVIEW

User Query: "{user_query}"
Classification (Return ONLY one word: FAQ, RECOMMENDATION, PRODUCT, or REVIEW):"""

        success, response = safe_chat(model=self.llm_model, prompt=prompt, temperature=0.0)

        if success:
            cleaned = response.strip().upper()
            for intent in ["RECOMMENDATION", "PRODUCT", "REVIEW", "FAQ"]:
                if intent in cleaned:
                    return intent

        # Secondary fallback heuristic
        if any(w in q_clean for w in ["price", "brand", "laptop", "mobile", "phone", "specs"]):
            return "PRODUCT"
        return "FAQ"


def main():
    detector = IntentDetector()
    test_queries = [
        "Where is my parcel?",
        "How do I return a product?",
        "Recommend a gaming laptop",
        "Suggest a budget smartphone",
        "Tell me about Lenovo Laptop",
        "What is its price?",
        "What brand is it?",
        "Summarize Lenovo customer reviews",
        "Is Lenovo Laptop worth buying?"
    ]

    print("=" * 60)
    print(" INTENT DETECTION VERIFICATION (Q12)")
    print("=" * 60)
    for q in test_queries:
        intent = detector.detect_intent(q)
        print(f"Query : '{q}'")
        print(f"Intent: {intent}")
        print("-" * 60)


if __name__ == "__main__":
    main()
