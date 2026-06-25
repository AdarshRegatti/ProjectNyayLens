# src/summarization/inference.py
import sys
import os
import re
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.summarization.ranker import ImportanceRanker
from src.summarization.utils import split_sentences
from src.segmentation.judgement_segmenter import JudgmentSegmenter
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "nsi319/legal-pegasus"
print(f"\nLoading Abstractive Model ({MODEL_NAME})...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print(f"✓ Legal-PEGASUS loaded on {device.upper()}")


def _pegasus_generate(text: str, max_length: int = 300, min_length: int = 100) -> str:
    """Run Legal-PEGASUS on a block of text and return the decoded summary."""
    inputs = pegasus_tokenizer(
        [text],
        max_length=1024,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    outputs = pegasus_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,                # Reduced from 8 for 2x speedup on CPU
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        early_stopping=True,
    )
    decoded = pegasus_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return re.sub(r'\s+', ' ', decoded.replace("<n>", " ")).strip()


def summarize(judgment_file: str) -> dict:
    """
    Speed-Optimized Two-Pass Pipeline:
    1. Case Overview: Legal-BERT (Extraction) -> Legal-PEGASUS (Abstraction) [1 Pass]
    2. Detailed Sections: Legal-BERT (Extraction) -> Direct Output [No Abstraction pass to save 5+ minutes]
    """
    text = Path(judgment_file).read_text(encoding="utf-8", errors="ignore")

    # ── Step 1: Global sentence extraction ─────────────────────────────────────
    all_sentences = [s for s in split_sentences(text) if len(s.strip()) > 40]
    if not all_sentences:
        return {"overview": "Could not extract readable text."}

    ranker = ImportanceRanker("outputs/summarization/final")
    scores = ranker.score(all_sentences)

    # Token-Aware Global Overview Extract (Limit to ~950 tokens for Pegasus)
    indexed = list(enumerate(zip(all_sentences, scores)))
    sorted_by_score = sorted(indexed, key=lambda x: x[1][1], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    MAX_TOKENS = 950
    
    for idx, (sentence, score) in sorted_by_score:
        tokens = len(pegasus_tokenizer.encode(sentence, add_special_tokens=False))
        if current_tokens + tokens > MAX_TOKENS:
            continue
            
        selected_indices.append(idx)
        current_tokens += tokens
        if current_tokens >= MAX_TOKENS - 20:
            break
            
    # Restore chronological order
    top_in_order = sorted([indexed[i] for i in selected_indices], key=lambda x: x[0])
    global_extract = " ".join(s for _, (s, _) in top_in_order)

    # ── Pass 1: Abstractive Overview (The only heavy pass) ────────────────────
    print("Generating Case Overview (Abstractive)...")
    overview = _pegasus_generate(global_extract, max_length=250, min_length=80)

    # ── Pass 2: Extractive Section Breakdown (Instant) ────────────────────────
    segmenter = JudgmentSegmenter()
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    sections = segmenter.segment(paragraphs)

    final_summary = {"overview": overview}

    print("Generating Section Breakdowns (Extractive - Instant)...")
    for section in sections:
        sec_type = section.type.lower()
        if sec_type == 'unknown': continue

        sentences = [s for s in split_sentences(section.text) if len(s.strip()) > 40]
        if not sentences: continue

        sec_scores = ranker.score(sentences)
        # Select top 3 per section for readability
        s_indexed = list(enumerate(zip(sentences, sec_scores)))
        top_k = sorted(s_indexed, key=lambda x: x[1][1], reverse=True)[:3]
        top_k_ordered = sorted(top_k, key=lambda x: x[0])
        
        # We use original sentences here to save 5-10 minutes of CPU time
        final_summary[sec_type] = " ".join(s for _, (s, _) in top_k_ordered)

    return final_summary


if __name__ == "__main__":
    file = list(Path("data/processed/extracted/texts").glob("*.txt"))[0]
    print(f"\nProcessing {file.name}...")
    result = summarize(file)

    print("\n\nCOMPREHENSIVE LEGAL SUMMARY (Global Legal-BERT + Legal-PEGASUS)\n" + "=" * 80)
    print("\n[CASE OVERVIEW]")
    print(result.get("overview", ""))
    for sec in ['facts', 'issues', 'arguments', 'analysis', 'decision']:
        if sec in result:
            print(f"\n[{sec.upper()}]")
            print(result[sec])
    print("\n" + "=" * 80)
