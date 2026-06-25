# """
# Annotate paragraphs with legal sections using JudgmentSegmenter
# Creates paragraph_index_with_sections.jsonl
# """

# import json
# from pathlib import Path
# from collections import defaultdict
# from tqdm import tqdm

# from judgement_segmenter import JudgmentSegmenter


# INPUT_INDEX = Path("data/processed/indexed/paragraph_index.jsonl")
# OUTPUT_INDEX = Path("data/processed/indexed/paragraph_index_with_sections.jsonl")


# def annotate_paragraphs():
#     print("=" * 70)
#     print("NyayLens – Annotating Paragraphs with Sections")
#     print("=" * 70)

#     # Load paragraphs grouped by judgment
#     judgments = defaultdict(list)

#     with open(INPUT_INDEX, "r", encoding="utf-8") as f:
#         for line in f:
#             p = json.loads(line)
#             judgments[p["judgment_id"]].append(p)

#     print(f"✓ Loaded {len(judgments):,} judgments")

#     segmenter = JudgmentSegmenter()

#     with open(OUTPUT_INDEX, "w", encoding="utf-8") as writer:
#         for judgment_id, paras in tqdm(judgments.items(), desc="Annotating"):
#             # Preserve original order
#             paras = sorted(paras, key=lambda x: (x["page_no"], x["id"]))

#             texts = [p["text"] for p in paras]

#             sections = segmenter.segment(texts)

#             # Default all to unknown
#             section_labels = [
#                 ("unknown", 0.0) for _ in paras
#             ]

#             # Apply section labels
#             for sec in sections:
#                 for i in range(sec.start_para_idx, sec.end_para_idx + 1):
#                     section_labels[i] = (sec.type, sec.confidence)

#             # Write annotated paragraphs
#             for p, (sec_type, sec_conf) in zip(paras, section_labels):
#                 p_out = dict(p)
#                 p_out["section"] = sec_type
#                 p_out["section_conf"] = sec_conf

#                 writer.write(json.dumps(p_out, ensure_ascii=False) + "\n")

#     print("\n✓ Annotation complete")
#     print(f"✓ Output written to: {OUTPUT_INDEX}")


# if __name__ == "__main__":
#     annotate_paragraphs()
"""
Annotate paragraphs with legal sections using JudgmentSegmenter
PRESERVES ORIGINAL IDs AND ORDER
"""

import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from judgement_segmenter import JudgmentSegmenter


INPUT_INDEX = Path("data/processed/indexed/paragraph_index.jsonl")
OUTPUT_INDEX = Path("data/processed/indexed/paragraph_index_with_sections.jsonl")


def annotate_paragraphs():
    print("=" * 70)
    print("NyayLens – Annotating Paragraphs with Sections")
    print("=" * 70)

    # Load paragraphs IN ORIGINAL ORDER
    all_paragraphs = []
    with open(INPUT_INDEX, "r", encoding="utf-8") as f:
        for line in f:
            all_paragraphs.append(json.loads(line))
    
    print(f"✓ Loaded {len(all_paragraphs):,} paragraphs")

    # Group by judgment (preserve index in group)
    judgments = defaultdict(list)
    for idx, p in enumerate(all_paragraphs):
        judgments[p["judgment_id"]].append((idx, p))  # ← Store original index

    segmenter = JudgmentSegmenter()
    
    # Create array to store annotations (preserves original order)
    annotations = [None] * len(all_paragraphs)

    for judgment_id, indexed_paras in tqdm(judgments.items(), desc="Annotating"):
        # Extract just the paragraphs
        indices = [ip[0] for ip in indexed_paras]
        paras = [ip[1] for ip in indexed_paras]
        
        # Get texts
        texts = [p["text"] for p in paras]

        # Segment
        sections = segmenter.segment(texts)

        # Default labels
        section_labels = [("unknown", 0.0) for _ in paras]

        # Apply section labels
        for sec in sections:
            for i in range(sec.start_para_idx, sec.end_para_idx + 1):
                if i < len(section_labels):
                    section_labels[i] = (sec.type, sec.confidence)

        # Store annotations in ORIGINAL positions
        for orig_idx, p, (sec_type, sec_conf) in zip(indices, paras, section_labels):
            p_out = dict(p)  # Copy original
            p_out["section"] = sec_type
            p_out["section_conf"] = sec_conf
            annotations[orig_idx] = p_out

    # Write in ORIGINAL order
    print("\nWriting annotated paragraphs...")
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as writer:
        for p_out in annotations:
            writer.write(json.dumps(p_out, ensure_ascii=False) + "\n")

    print(f"✓ Output written to: {OUTPUT_INDEX}")
    print("=" * 70)


if __name__ == "__main__":
    annotate_paragraphs()
