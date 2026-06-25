"""
Paragraph-level indexer for NyayLens RAG
- Page-aware
- Content-stable paragraph IDs
- Legal-aware filtering
- Streaming JSONL output (memory safe)
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable
from tqdm import tqdm


class ParagraphIndexer:
    """Index legal judgments at paragraph level with stable IDs and metadata"""

    # Legal keywords worth preserving even in short paragraphs
    LEGAL_KEYWORDS = {
        'held', 'order', 'appeal', 'writ', 'judgment', 'decree',
        'petition', 'application', 'allowed', 'dismissed', 'granted',
        'rejected', 'disposed', 'quashed', 'set aside',
        'affirmed', 'reversed', 'remanded', 'bail', 'custody',
        'interim', 'stay', 'injunction', 'no costs'
    }

    PAGE_MARKER_PATTERN = re.compile(r'<<<PAGE:(\d+)>>>')

    def __init__(self, texts_dir: Path, output_dir: Path):
        self.texts_dir = Path(texts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.output_dir / "paragraph_index.jsonl"
        self.stats_file = self.output_dir / "index_stats.json"

    @staticmethod
    def _contains_legal_keyword(text: str) -> bool:
        text_l = text.lower()
        return any(
            re.search(rf"\b{re.escape(kw)}\b", text_l)
            for kw in ParagraphIndexer.LEGAL_KEYWORDS
        )

    @staticmethod
    def _stable_paragraph_id(judgment_id: str, page_no: int, text: str) -> str:
        """
        Content-stable ID:
        - same paragraph text => same ID
        - survives re-indexing
        """
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        page_str = page_no if page_no is not None else "unk"
        return f"{judgment_id}_p{page_str}_{h}"

    def _strip_header(self, content: str) -> str:
        """
        Remove extractor quality header safely
        """
        sep = "=" * 70
        if sep in content:
            parts = content.split(sep, 2)
            if len(parts) == 3:
                return parts[2].strip()
        return content.strip()

    def _iter_paragraphs(self, content: str) -> Iterable[tuple]:
        """
        Yield paragraph records with page numbers
        """
        current_page = None
        buffer = []

        for line in content.splitlines():
            page_match = self.PAGE_MARKER_PATTERN.match(line.strip())
            if page_match:
                # Flush buffer before page change
                if buffer:
                    yield current_page, "\n".join(buffer).strip()
                    buffer = []
                current_page = int(page_match.group(1))
                continue

            if not line.strip():
                if buffer:
                    yield current_page, "\n".join(buffer).strip()
                    buffer = []
                continue

            buffer.append(line)

        if buffer:
            yield current_page, "\n".join(buffer).strip()

    def index_judgment(self, text_file: Path, writer) -> int:
        """
        Index a single judgment file.
        Returns number of paragraphs indexed.
        """
        with open(text_file, "r", encoding="utf-8") as f:
            content = self._strip_header(f.read())

        judgment_id = text_file.stem
        para_count = 0

        for page_no, para in self._iter_paragraphs(content):
            if not para:
                continue

            # Keep substantial OR legally important short paragraphs
            if len(para) < 50 and not self._contains_legal_keyword(para):
                continue

            record = {
                "id": self._stable_paragraph_id(judgment_id, page_no if page_no is not None else -1, para),
                "judgment_id": judgment_id,
                "page_no": page_no if page_no is not None else -1,
                "text": para,
                "char_count": len(para),
                "word_count": len(para.split())
            }

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            para_count += 1

        return para_count

    def build_full_index(self):
        text_files = sorted(self.texts_dir.glob("*.txt"))
        print(f"Indexing {len(text_files):,} judgments...")

        total_paragraphs = 0

        with open(self.index_file, "w", encoding="utf-8") as writer:
            for text_file in tqdm(text_files, desc="Indexing"):
                try:
                    total_paragraphs += self.index_judgment(text_file, writer)
                except Exception as e:
                    print(f"❌ Failed indexing {text_file.name}: {e}")

        stats = {
            "total_judgments": len(text_files),
            "total_paragraphs": total_paragraphs,
            "avg_paragraphs_per_judgment":
                total_paragraphs / len(text_files) if text_files else 0
        }

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print("\n✓ Paragraph indexing complete")
        print(f"  Total paragraphs: {total_paragraphs:,}")
        print(f"  Output: {self.index_file}")

        return stats


if __name__ == "__main__":
    indexer = ParagraphIndexer(
        texts_dir=Path("data/processed/extracted/texts"),
        output_dir=Path("data/processed/indexed")
    )

    stats = indexer.build_full_index()
    print(f"\nAverage paragraphs per judgment: {stats['avg_paragraphs_per_judgment']:.1f}")
