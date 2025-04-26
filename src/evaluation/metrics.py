import time
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

class Evaluator:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.rouge = Rouge()
        
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