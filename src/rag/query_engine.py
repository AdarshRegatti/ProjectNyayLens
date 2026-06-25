"""RAG Query Engine with LLM"""

# pyrefly: ignore [missing-import]
import faiss
import json
import sqlite3
import re
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from groq import Groq
from dotenv import load_dotenv

from src.summarization.ranker import ImportanceRanker
from src.summarization.utils import split_sentences

load_dotenv()

class QueryEngine:
    
    def __init__(self):
        print("Loading RAG components...")
        
        # FAISS + SQLite + Embeddings
        self.index = faiss.read_index("data/processed/faiss/faiss_index.bin")
        
        with open("data/processed/embeddings/paragraph_ids.json") as f:
            self.para_ids = json.load(f)
        
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        self.importance_ranker = ImportanceRanker("outputs/summarization/final")
        
    def _get_db(self):
        return sqlite3.connect("data/processed/indexed/paragraphs.db")
        
        # LLM Setup
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        self.llm = Groq(api_key=api_key)
        self.llm_model = 'llama-3.1-8b-instant'  
        
        print(f"✓ Ready with {self.index.ntotal:,} vectors")
        print(f"✓ LLM: Groq (Llama 3.1 8B)")
    
    def search(self, query: str, top_k: int = 5):
        """Hybrid Search: FAISS (Dense) + SQLite FTS5 (BM25) with RRF"""
        
        # --- 1. Dense Search (FAISS) ---
        query_vec = self.model.encode([query], normalize_embeddings=True)
        dense_scores, dense_indices = self.index.search(query_vec, k=top_k * 2) # Fetch extra for fusion
        
        dense_results = []
        for rank, (score, idx) in enumerate(zip(dense_scores[0], dense_indices[0])):
            para_id = self.para_ids[idx]
            dense_results.append({'id': para_id, 'score': float(score), 'rank': rank + 1})
            
        # --- 2. Keyword Search (SQLite FTS5 BM25) ---
        db = self._get_db()
        cursor = db.cursor()
        
        # FTS5 requires a specific syntax. A raw string means AND.
        # We want BM25 behavior (OR), so we clean punctuation and join words with OR.
        import re
        clean_query = re.sub(r'[^\w\s]', '', query)
        fts_query = " OR ".join(clean_query.split())
        
        try:
            cursor.execute(f"""
                SELECT id, bm25(paragraphs_fts) as bm25_score
                FROM paragraphs_fts 
                WHERE paragraphs_fts MATCH ?
                ORDER BY bm25_score LIMIT ?
            """, (fts_query, top_k * 2))
            fts_rows = cursor.fetchall()
            
            keyword_results = []
            for rank, row in enumerate(fts_rows):
                keyword_results.append({'id': row[0], 'score': float(row[1]), 'rank': rank + 1})
        except sqlite3.OperationalError:
            # Fallback if query syntax is too complex for basic MATCH or FTS table missing
            keyword_results = []

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        # RRF Score = 1 / (k + rank)  where k is usually 60
        k = 60
        rrf_scores = {}
        
        # Add dense scores
        for res in dense_results:
            pid = res['id']
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + res['rank']))
            
        # Add keyword scores
        for res in keyword_results:
            pid = res['id']
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (k + res['rank']))
            
        # Sort by RRF score descending
        # Fetch a larger pool of candidates for reranking
        candidate_pool_size = top_k * 3
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_pool_size]
        
        # --- 4. Fetch Details & Rerank (Cross-Encoder) ---
        candidates = []
        for pid, rrf_score in sorted_rrf:
            cursor.execute(
                "SELECT judgment_id, text, page_no FROM paragraphs WHERE id = ?",
                (pid,)
            )
            row = cursor.fetchone()
            
            if row:
                candidates.append({
                    'rrf_score': rrf_score,
                    'judgment_id': row[0],
                    'text': row[1],
                    'page_no': row[2],
                    'id': pid
                })
        
        if not candidates:
            db.close()
            return []
            
        db.close()

        # --- 5. Final Rerank (Cross-Encoder) ---
            
        # Prepare inputs for cross-encoder: list of [query, document_text]
        cross_inp = [[query, doc['text']] for doc in candidates]
        rerank_scores = self.reranker.predict(cross_inp)
        
        # Attach scores and sort
        for i, score in enumerate(rerank_scores):
            candidates[i]['score'] = float(score)  # Use cross-encoder score as final score
            
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]
    
    def generate_answer(self, question: str, context: str, sources: list = [], chat_history: list = None):
        """Generate answer using Groq LLM with strict Legal Guardrails.
        Sources list is injected into the prompt so the LLM can ONLY cite
        what was actually retrieved — no hallucinated references.
        """
        chat_history = chat_history or []
        chat_history = chat_history[-6:]  # Cap to last 3 turns
        # Build a numbered source registry for the LLM
        source_registry = ""
        for i, s in enumerate(sources, 1):
            source_registry += f"[{i}] {s.get('judgment_id', 'Unknown')}\n"

        prompt = f"""You are a strict, brilliant legal research assistant specializing in Indian Supreme Court judgments.

GUARDRAIL: You MUST ONLY answer questions related to law, legal processes, or the provided context.
If the question is entirely unrelated to law (e.g., "how to bake a cake"), reply EXACTLY with:
"I am a legal AI assistant. I can only answer questions related to law."

CITATION RULES — THIS IS CRITICAL:
- You may ONLY cite sources from the APPROVED SOURCE LIST below.
- Do NOT cite any case from your training memory that is not in the APPROVED SOURCE LIST.
- If you cite a case not in this list, you are hallucinating and failing your task.
- Use [1], [2], [3] etc. to refer to sources from the list below.

APPROVED SOURCE LIST (cite ONLY these):
{source_registry}
CONTEXT (retrieved paragraphs):
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a detailed, comprehensive legal answer in a professional conversational tone.
- Explain concepts clearly so a lawyer finds it extremely useful.
- Cite ONLY from the APPROVED SOURCE LIST above using [1], [2], [3] format.
- Use proper legal terminology.
- Do NOT invent case names, citations, or dates.
- TEMPORAL AWARENESS: Look at the years in the judgment titles (e.g. 2023_CaseName). Newer judgments (e.g. 2023) supersede older judgments (e.g. 2010). If the retrieved context contains conflicting rulings, you MUST prioritize the newer judgment and explicitly warn the user that the older precedent may have been superseded.

ANSWER:"""

        messages = chat_history.copy()
        messages.append({"role": "user", "content": prompt})

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 5, chat_history: list = None):
        """Main query method"""
        print(f"\n{'='*70}")
        print(f"QUERY: {question}")
        print('='*70)
        
        # Search
        print("\nSearching FAISS index...")
        results = self.search(question, top_k)
        
        print(f"Found {len(results)} relevant paragraphs")
        
        # Format context
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {r['judgment_id']}\n{r['text']}"
            )
        context = "\n\n".join(context_parts)
        
        # Generate answer — pass sources so LLM can only cite what was retrieved
        print("Generating answer with LLM...")
        answer = self.generate_answer(question, context, sources=results, chat_history=chat_history)
        
        return {
            'question': question,
            'answer': answer,
            'sources': results
        }
        
    def query_with_document(self, question: str, filepath: str, chat_history: list = None):
        """Queries a specific document. Falls back to global RAG if answer not found."""
        chat_history = chat_history or []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                doc_text = f.read()
        except Exception as e:
            return {"answer": f"Error reading document: {e}", "sources": []}
            
        # JUDGMENTS are usually under 30k chars. Let's take as much as possible.
        if len(doc_text) > 30000:
            print("Document exceeds 30k chars. Applying Semantic Truncation...")
            try:
                sentences = [s for s in split_sentences(doc_text) if len(s.strip()) > 20]
                scores = self.importance_ranker.score(sentences)
                indexed = list(enumerate(zip(sentences, scores)))
                sorted_by_score = sorted(indexed, key=lambda x: x[1][1], reverse=True)
                
                selected_indices = []
                current_chars = 0
                for idx, (sentence, score) in sorted_by_score:
                    if current_chars + len(sentence) > 30000:
                        continue
                    selected_indices.append(idx)
                    current_chars += len(sentence)
                    if current_chars > 29000:
                        break
                
                # Restore original chronological order
                top_in_order = sorted([indexed[i] for i in selected_indices], key=lambda x: x[0])
                doc_text = " ".join(s for _, (s, _) in top_in_order) + "\n\n... [TRUNCATED SEMANTICALLY FOR LLM] ..."
            except Exception as e:
                print(f"Semantic Truncation failed: {e}. Falling back to naive truncation.")
                doc_text = doc_text[:30000] + "\n\n... [TRUNCATED DUE TO SIZE] ..."
            
        print(f"--- DOCUMENT QA START ---")
        print(f"File: {os.path.basename(filepath)}")
        print(f"Size: {len(doc_text)} chars")
        print(f"Question: {question}")

        prompt = f"""You are a strict Legal Document Auditor.
Your ONLY source of information is the text provided below. 

STRICT RULES:
1. Answer the QUESTION using ONLY the DOCUMENT text.
2. If the answer is not in the text, say "I cannot find this in the uploaded document."
3. DO NOT cite external cases (like Venkata Reddy or V.C. Shukla) unless they are explicitly mentioned in the text below.
4. If you use your own internal knowledge instead of the document, you are failing your task.

DOCUMENT TEXT:
{doc_text}

QUESTION: {question}

DETAILED ANSWER (citing specific paragraphs if possible):"""

        messages = chat_history.copy()
        messages.append({"role": "user", "content": prompt})

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024
        )
        
        answer = response.choices[0].message.content.strip()
            
        return {
            'question': question,
            'answer': answer,
            'sources': [{'judgment_id': os.path.basename(filepath), 'score': 1.0}]
        }
    
    def close(self):
        self.db.close()

# Test
if __name__ == "__main__":
    engine = QueryEngine()
    
    # Test queries
    queries = [
        "What are the conditions for granting anticipatory bail?",
        "Explain the doctrine of legitimate expectation",
        "What is the burden of proof in criminal cases?"
    ]
    
    for query in queries:
        response = engine.query(query, top_k=3)
        
        print(f"\nANSWER:\n{response['answer']}\n")
        
        print("SOURCES:")
        for i, src in enumerate(response['sources'], 1):
            print(f"  [{i}] {src['judgment_id']} (score: {src['score']:.3f})")
        
        print("\n" + "="*70 + "\n")
    
    engine.close()
