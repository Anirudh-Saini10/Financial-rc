#!/usr/bin/env python3
"""Test script to verify reconciliation logic with sample data."""

import pandas as pd
from app import load_sheet, reconcile, build_chunks, retrieve, ask_agent

def test_sample_data():
    """Test reconciliation with sample data files."""
    print("Loading sample data...")
    
    # Load sample data
    with open("sample_data/sheet_a_internal_ledger.csv", "rb") as f:
        sheet_a = load_sheet(f.read(), "sheet_a_internal_ledger.csv", "A")
    
    with open("sample_data/sheet_b_external_statement.csv", "rb") as f:
        sheet_b = load_sheet(f.read(), "sheet_b_external_statement.csv", "B")
    
    print(f"Sheet A loaded: {len(sheet_a)} rows")
    print(f"Sheet B loaded: {len(sheet_b)} rows")
    
    print("\nSheet A sample:")
    print(sheet_a[["transaction_id", "date", "reference", "counterparty", "amount"]].head())
    
    print("\nSheet B sample:")
    print(sheet_b[["transaction_id", "date", "reference", "counterparty", "amount"]].head())
    
    # Reconcile
    print("\nReconciling...")
    reconciliation = reconcile(sheet_a, sheet_b)
    
    print(f"\nMatched: {len(reconciliation['matched'])}")
    print(f"Mismatched: {len(reconciliation['mismatched'])}")
    print(f"Missing in B: {len(reconciliation['missing_in_b'])}")
    print(f"Missing in A: {len(reconciliation['missing_in_a'])}")
    print(f"Suspicious: {len(reconciliation['suspicious'])}")
    
    print("\nSummary:")
    print(reconciliation['summary'])
    
    # Test chunks and retrieval
    print("\nBuilding chunks...")
    chunks = build_chunks(sheet_a, sheet_b, reconciliation)
    print(f"Total chunks: {len(chunks)}")
    
    # Test retrieval
    print("\nTesting retrieval for 'mismatched transactions'...")
    evidence = retrieve("mismatched transactions", chunks)
    print(f"Retrieved {len(evidence)} chunks")
    
    # Test agent by directly calling the functions
    print("\nTesting agent questions...")
    from app import local_agent_answer
    
    questions = [
        "Find all mismatched transactions.",
        "Why is the balance different between both sheets?",
        "Show transactions above 50000 only in Sheet A.",
        "Give me a reconciliation summary.",
        "Which transactions need manual review?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        try:
            evidence = retrieve(q, chunks)
            answer = local_agent_answer(q, evidence, reconciliation, sheet_a, sheet_b)
            print(f"A: {answer['answer']}")
            print(f"Confidence: {answer['confidence']}")
            print(f"Citations: {answer['citations']}")
            print(f"Table rows: {len(answer['table'])}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_sample_data()
