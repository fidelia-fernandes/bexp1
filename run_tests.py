import os
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ECommerceGPT"))
from chatbot import ECommerceChatbot

bot = ECommerceChatbot()

test_queries = [
    "Where is my parcel?",
    "Recommend a gaming laptop",
    "Tell me about Samsung Galaxy",
    "What is its price?",
    "What are customer reviews for Lenovo Laptop?",
    "exit"
]

print("\n--- BEGIN INTEGRATED BOT TEST ---\n")
for q in test_queries:
    print(f"You :\n{q}")
    intent, reply = bot.handle_query(q)
    print("Assistant")
    print(reply)
    print()
    if intent == "EXIT":
        break
print("--- END INTEGRATED BOT TEST ---")
