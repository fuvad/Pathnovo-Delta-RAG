"""
Interactive CLI Chat — ask questions about your document revisions and delta report.
"""

import sys
from pathlib import Path
from src.ingest.pdf_native import NativePDFAdapter
from src.delta.engine import DeltaEngine
from src.delta.report import generate_report
from src.chat.index import QdrantIndexer
from src.chat.answer import GroundedChat
from src.config.logging import setup_logging

def main():
    setup_logging()
    
    pdf_a = Path("data/samples/Export Gas Compressor-P&ID.pdf")
    pdf_b = Path("data/samples/Lift Gas compressor-P&ID.pdf")
    pid_a = "export_gas_902"
    pid_b = "lift_gas_901"

    print("\n" + "="*60)
    print("  DOCUMENT DELTA & GROUNDED CHAT CLI")
    print("="*60)
    print("  Loading and processing documents...")
    
    # 1. Ingest
    adapter = NativePDFAdapter()
    doc_a = adapter.ingest(pdf_a, pid_a)
    doc_b = adapter.ingest(pdf_b, pid_b)
    
    # 2. Delta
    engine = DeltaEngine()
    deltas = engine.compute_delta(doc_a, doc_b)
    generate_report(pid_a, pid_b, deltas)
    
    # 3. Index into Qdrant
    print("  Indexing documents into Qdrant vector database...")
    indexer = QdrantIndexer()
    indexer.index_document(doc_a)
    indexer.index_document(doc_b)
    
    # 4. Chat Setup
    chat = GroundedChat()
    chat.load_delta(pid_a, pid_b, deltas)
    
    print("\n  Ready! Type your question below (or type 'exit' or 'quit' to stop).\n")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n\033[1;36mAsk a question > \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chat. Goodbye!")
                break
                
            print("\nSearching context and generating grounded response...\n")
            result = chat.ask(user_input)
            
            print("\033[1;32mAnswer:\033[0m")
            print(result["answer"])
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
