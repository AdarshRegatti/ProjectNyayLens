import torch
import faiss
import json
import sqlite3
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class LegalQAEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # ---- Load QA model ----
        self.tokenizer = AutoTokenizer.from_pretrained("outputs/qa_model/final")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "outputs/qa_model/final"
        ).to(self.device)
        self.qa_model.eval()

        # ---- Load retriever ----
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=self.device)
        self.index = faiss.read_index("data/processed/faiss/faiss_index.bin")

        with open("data/processed/embeddings/paragraph_ids.json", encoding="utf-8") as f:
            self.para_ids = json.load(f)

        self.db = sqlite3.connect("data/processed/indexed/paragraphs.db")
        self.cursor = self.db.cursor()

        print("✓ Enhanced QA inference system ready")

    # ------------------------------------------------------------------
    # TEXT NORMALIZATION (critical for PDF artifacts)
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # REFUTED CLAUSE DETECTION (Article 21 FIX)
    # ------------------------------------------------------------------
    def _is_refuted_clause(self, answer_text, paragraph_text):
        para = self._normalize(paragraph_text)
        ans = self._normalize(answer_text)

        # Patterns like:
        # "it is not correct to say, ..., that X"
        # "it cannot be said, ..., that X"
        refutation_regexes = [
            r"not correct to say.*?that\s+(.*?)(?:\.|,)",
            r"cannot be said.*?that\s+(.*?)(?:\.|,)",
        ]

        for pattern in refutation_regexes:
            matches = re.findall(pattern, para)
            for refuted_prop in matches:
                # If answer is part of the refuted proposition → block
                if ans in refuted_prop:
                    return True

        return False


    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------
    def retrieve_paragraphs(self, question, top_k=8):
        q_emb = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            para_id = self.para_ids[idx]
            self.cursor.execute(
                "SELECT judgment_id, page_no, text FROM paragraphs WHERE id = ?",
                (para_id,),
            )
            row = self.cursor.fetchone()
            if row:
                judgment_id, page_no, text = row
                results.append(
                    {
                        "judgment_id": judgment_id,
                        "page_no": page_no,
                        "text": text,
                        "retrieval_score": float(score),
                    }
                )
        return results

    # ------------------------------------------------------------------
    # ANSWERING
    # ------------------------------------------------------------------
    def answer_question(self, question, top_k=8, max_answers=2):
        paragraphs = self.retrieve_paragraphs(question, top_k)
        candidates = []

        for para in paragraphs:
            inputs = self.tokenizer(
                question,
                para["text"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            token_type_ids = inputs["token_type_ids"][0].tolist()
            question_end = token_type_ids.index(1)

            top_starts = torch.topk(start_logits, k=5).indices
            top_ends = torch.topk(end_logits, k=5).indices

            for s in top_starts:
                for e in top_ends:
                    if e < s or (e - s) > 80:
                        continue

                    # ❌ Block question echo
                    if s < question_end:
                        continue

                    answer_tokens = inputs["input_ids"][0][s : e + 1]
                    answer_text = self.tokenizer.decode(
                        answer_tokens, skip_special_tokens=True
                    ).strip()

                    words = answer_text.split()
                    if len(words) < 8:
                        continue

                    # ❌ Block refuted propositions
                    if self._is_refuted_clause(answer_text, para["text"]):
                        continue

                    score = start_logits[s].item() + end_logits[e].item()

                    # Doctrinal boost
                    if any(
                        k in answer_text.lower()
                        for k in ["the court", "held that", "it is clear that", "the law"]
                    ):
                        score += 1.5

                    candidates.append(
                        {
                            "answer": answer_text,
                            "confidence": score,
                            "judgment_id": para["judgment_id"],
                            "page_no": para["page_no"],
                            "paragraph": para["text"],
                            "retrieval_score": para["retrieval_score"],
                        }
                    )

        # ---- Deduplicate answers ----
        seen = set()
        final = []
        for c in sorted(candidates, key=lambda x: x["confidence"], reverse=True):
            key = self._normalize(c["answer"])
            if key not in seen:
                seen.add(key)
                final.append(c)

        return final[:max_answers]


# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------
if __name__ == "__main__":
    qa = LegalQAEngine()

    questions = [
        "What is the scope of Article 21?",
        "What are the conditions for granting anticipatory bail?",
        "What is the burden of proof in criminal law?",
    ]

    for q in questions:
        print("\n" + "=" * 90)
        print(f"QUESTION: {q}")
        print("=" * 90)

        answers = qa.answer_question(q)

        for i, ans in enumerate(answers, 1):
            print(f"\nANSWER {i}:")
            print(ans["answer"])
            print(
                f"\nSOURCE: {ans['judgment_id']} | Page: {ans['page_no']}"
            )
            print(f"Retrieval score: {ans['retrieval_score']:.3f}")
            print(f"Confidence score: {ans['confidence']:.2f}")
            print("\nPARAGRAPH:")
            print(ans["paragraph"][:700] + "...")
