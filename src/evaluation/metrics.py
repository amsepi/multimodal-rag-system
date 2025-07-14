import time
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from collections import defaultdict

class Evaluator:
    def __init__(self, retriever, generator, vector_db=None):
        self.retriever = retriever
        self.generator = generator
        self.rouge = Rouge()
        self.vector_db = vector_db
    
    def chunking_stats(self):
        """Return stats on text chunks in the vector DB."""
        if not self.vector_db:
            return {}
        metadatas = self.vector_db.text_collection.get()["metadatas"]
        doc_stats = defaultdict(lambda: {"chunks": 0, "table_chunks": 0, "min_len": float('inf'), "max_len": 0, "total_len": 0})
        for meta in metadatas:
            doc = meta["source"]
            l = meta.get("chunk_length", 0)
            doc_stats[doc]["chunks"] += 1
            doc_stats[doc]["table_chunks"] += int(meta.get("has_table", False))
            doc_stats[doc]["min_len"] = min(doc_stats[doc]["min_len"], l)
            doc_stats[doc]["max_len"] = max(doc_stats[doc]["max_len"], l)
            doc_stats[doc]["total_len"] += l
        # Finalize
        for doc, stats in doc_stats.items():
            if stats["chunks"]:
                stats["avg_len"] = stats["total_len"] / stats["chunks"]
            else:
                stats["avg_len"] = 0
        return doc_stats

    def embedding_stats(self):
        """Return stats on embedding failures (if any)."""
        # This would require logging failures in add_documents; for now, assume all succeeded.
        # Could be extended to log failures in VectorStore.
        return {}

    def retrieval_benchmarks(self, test_questions, top_ks=[1,3,5]):
        """Return per-question and aggregate retrieval accuracy for various top-k."""
        results = []
        hit_counts = {k: 0 for k in top_ks}
        for q in test_questions:
            if isinstance(q, dict):
                query = q["question"]
                gt_doc = q["ground_truth_doc"]
            else:
                query = q[0]
                gt_doc = q[1]
            # Ensure query is a string
            if not isinstance(query, str):
                print(f"[Evaluator] Skipping non-string query: {query}")
                continue
            if not query.strip():
                print(f"[Evaluator] Skipping empty query string.")
                continue
            retrieved = self.retriever.retrieve(query, top_k=max(top_ks))
            retrieved_docs = [res["metadata"]["source"] for res in retrieved]
            row = {"question": query, "ground_truth": gt_doc}
            for k in top_ks:
                row[f"hit@{k}"] = gt_doc in retrieved_docs[:k]
                hit_counts[k] += int(row[f"hit@{k}"])
            row["retrieved_docs"] = retrieved_docs[:max(top_ks)]
            row["retrieved_chunks"] = [res["content"][:120] for res in retrieved[:max(top_ks)]]
            row["scores"] = [res["score"] for res in retrieved[:max(top_ks)]]
            results.append(row)
        # Aggregate
        agg = {f"hit@{k}": hit_counts[k]/len(test_questions) for k in top_ks}
        return results, agg

    def calculate_hit_rate(self, queries, ground_truths, top_k=5):
        """Calculate retrieval hit rate"""
        hits = 0
        for query, gt in zip(queries, ground_truths):
            results = self.retriever.retrieve(query, top_k)
            retrieved_docs = [res['metadata']['source'] for res in results]
            hits += 1 if gt in retrieved_docs else 0
        return hits / len(queries)
    
    def evaluate_response(self, response, reference):
        """Calculate BLEU and ROUGE scores"""
        bleu = sentence_bleu([reference.split()], response.split())
        rouge = self.rouge.get_scores(response, reference)[0]
        return {
            'bleu-4': bleu,
            'rouge-l': rouge['rouge-l']['f']
        }
    
    def measure_latency(self, query, num_runs=10):
        """Average response time measurement"""
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.generator.generate_response(query, self.retriever.retrieve(query))
            times.append(time.time() - start)
        return np.mean(times)