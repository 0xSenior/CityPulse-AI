import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.traffic_rag import TrafficRAG

def run_tests():
    print("=" * 60)
    print("🚦 TESTING NYC SMART TRAFFIC RAG PIPELINE")
    print("=" * 60)
    
    rag = TrafficRAG()
    
    queries = [
        "إيه حالة المرور في شارع Flatbush Avenue الساعة 6 مساءً؟",
        "هل فيه زحمة أو خطر حوادث عند Washington Bridge الصبح الساعة 8؟",
        "ما هي حالة السير في Hunts Point Avenue في البرونكس؟",
        "Traffic forecast for 94th Avenue in Queens at 5 PM",
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n[Test Query {i}]: {q}")
        res = rag.query(q)
        print(res["answer"])
        print("-" * 60)

if __name__ == "__main__":
    run_tests()
