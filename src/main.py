from rag_chain import build_rag_chain
rag_chain, chunk_count = build_rag_chain()
print(f"RAG chain ready with {chunk_count} chunks")

test_questions = [
    "What is the minimum capital adequency ration required for NBFC-MFIs?"
]

for q in test_questions:
    print(f"\n{'='*80}")
    print(f"QUESTION: {q}")
    print(f"{'='*80}")
    answer = rag_chain.invoke(q)
    print(f"\nANSWER:\n{answer}")