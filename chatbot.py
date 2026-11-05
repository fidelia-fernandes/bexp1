"""
chatbot.py
==========
Part F - Q13: Integrated AI-Powered E-Commerce Chatbot using Ollama.
Continuous interactive console chatbot combining:
- Intent Detection (Ollama)
- Semantic FAQ Search with cosine similarity & threshold (Ollama embeddings)
- Product Recommendation with TF-IDF and LLM justification (Ollama)
- Product Question Answering RAG with constrained prompt (Ollama)
- Customer Review Intelligence & Sentiment Analysis (VADER / TextBlob + Ollama)
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from intent_module import IntentDetector
from faq_module import FAQModule
from recommendation_module import RecommendationModule
from product_module import ProductQAModule
from review_module import ReviewIntelligenceModule
from ollama_client import check_ollama_daemon


class ECommerceChatbot:
    def __init__(self, llm_model="qwen2.5:0.5b", embed_model="nomic-embed-text"):
        print("[Chatbot] Initializing ECommerceGPT modules...")
        self.llm_model = llm_model
        self.embed_model = embed_model

        # Verify Ollama connectivity
        ok, msg = check_ollama_daemon()
        if not ok:
            print(f"[Chatbot Warning] {msg}")
            print("[Chatbot Warning] Ensure 'ollama serve' is running for full LLM generation.\n")

        self.intent_detector = IntentDetector(llm_model=llm_model)
        self.faq_module = FAQModule(threshold=0.50, embed_model=embed_model, llm_model=llm_model)
        self.rec_module = RecommendationModule(llm_model=llm_model)
        self.product_module = ProductQAModule(llm_model=llm_model)
        self.review_module = ReviewIntelligenceModule(llm_model=llm_model)
        print("[Chatbot] All modules loaded and operational.\n")

    def handle_query(self, user_query):
        """
        Process user query through intent detection and modular routing.
        Returns: (intent: str, response: str)
        """
        q_clean = user_query.strip()
        if not q_clean:
            return "EMPTY", "Please enter a valid question or product inquiry."

        intent = self.intent_detector.detect_intent(q_clean)

        print("↓")
        print("Detected Intent")
        print(intent)
        print("↓")

        if intent == "EXIT":
            return "EXIT", "Thank you for shopping with us! Have a wonderful day."

        # Module 1: FAQ
        elif intent == "FAQ":
            faq_res = self.faq_module.process_query(q_clean)
            if faq_res["matched"]:
                return intent, faq_res["response"]
            else:
                # If FAQ score is low, attempt product search fallback
                prod_res = self.product_module.answer_question(q_clean)
                return "FAQ (Fallback to Product QA)", prod_res["answer"]

        # Module 2: RECOMMENDATION
        elif intent == "RECOMMENDATION":
            rec_res = self.rec_module.process_query(q_clean, top_k=3)
            return intent, rec_res["recommendation"]

        # Module 3: PRODUCT
        elif intent == "PRODUCT":
            prod_res = self.product_module.answer_question(q_clean)
            return intent, prod_res["answer"]

        # Module 4: REVIEW
        elif intent == "REVIEW":
            rev_res = self.review_module.process_query(q_clean)
            analysis = rev_res["analysis"]
            summary_output = (
                f"Review Sentiment Breakdown for {analysis['product_name']}:\n"
                f"Positive : {analysis['positive']} | Neutral : {analysis['neutral']} | Negative : {analysis['negative']}\n\n"
                f"{rev_res['summary']}"
            )
            return intent, summary_output

        # Default fallback
        else:
            faq_res = self.faq_module.process_query(q_clean)
            return "FAQ", faq_res["response"]

    def run_interactive(self):
        """Run continuous interactive conversation loop until user enters Exit."""
        print("=" * 60)
        print("   AI-POWERED E-COMMERCE CHATBOT (Ollama / Qwen2.5)")
        print("=" * 60)
        print("Ask any question regarding FAQs, recommendations, products, or reviews.")
        print("Type 'exit' or 'quit' to terminate the session.\n")

        while True:
            try:
                user_input = input("You :\n").strip()
                if not user_input:
                    continue

                intent, reply = self.handle_query(user_input)

                if intent == "EXIT":
                    print("Assistant")
                    print(reply)
                    break

                print("Assistant")
                print(reply)
                print()

            except (KeyboardInterrupt, EOFError):
                print("\nAssistant")
                print("Session closed. Goodbye!")
                break


def main():
    bot = ECommerceChatbot()
    bot.run_interactive()


if __name__ == "__main__":
    main()
