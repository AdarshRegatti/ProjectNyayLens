import argparse
from pathlib import Path
import logging
import time

from extraction.pdf_extractor import LegalJudgmentExtractor
from segmentation.judgement_segmenter import JudgmentSegmenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NyayLensPipeline:
    """Unified ingestion pipeline for NyayLens"""
    
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
        logger.info("Initializing NyayLens Pipeline components...")
        self.extractor = LegalJudgmentExtractor(output_dir=self.processed_dir / "extracted")
        self.segmenter = JudgmentSegmenter()
        
    def ingest_pdf(self, pdf_path: Path):
        """Process a single PDF end-to-end"""
        logger.info(f"--- Starting Ingestion: {pdf_path.name} ---")
        start_time = time.time()
        
        # 1. Extraction
        logger.info("Step 1: Extracting text...")
        extraction_result = self.extractor.extract_pdf(pdf_path)
        if not extraction_result['text']:
            logger.error(f"Extraction failed for {pdf_path.name}")
            return False
            
        # Optional: Save extraction
        self.extractor.save_extraction(pdf_path, extraction_result)
        
        # 2. Segmentation
        logger.info("Step 2: Segmenting judgment...")
        # Simple paragraph split for segmentation
        paragraphs = extraction_result['text'].split('\n\n')
        sections = self.segmenter.segment(paragraphs)
        
        logger.info(f"Found {len(sections)} distinct sections.")
        
        # 3. Next steps would hook into `create_embeddings.py` and `create_sqlite_index.py`
        logger.info("Step 3: Ready for Embeddings & Indexing (Batch process recommended)")
        
        elapsed = time.time() - start_time
        logger.info(f"--- Successfully processed {pdf_path.name} in {elapsed:.2f}s ---")
        return True

    def process_directory(self, limit: int = None):
        """Process all PDFs in the raw directory"""
        pdfs = list(self.raw_dir.glob("**/*.pdf")) + list(self.raw_dir.glob("**/*.PDF"))
        
        if limit:
            pdfs = pdfs[:limit]
            
        logger.info(f"Found {len(pdfs)} PDFs to process.")
        
        success = 0
        for pdf in pdfs:
            if self.ingest_pdf(pdf):
                success += 1
                
        logger.info(f"Batch completed. {success}/{len(pdfs)} successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NyayLens Unified Ingestion Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF to ingest")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in data/raw")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs in batch")
    
    args = parser.parse_args()
    
    pipeline = NyayLensPipeline()
    
    if args.pdf:
        pipeline.ingest_pdf(Path(args.pdf))
    elif args.batch:
        pipeline.process_directory(limit=args.limit)
    else:
        logger.warning("Please specify --pdf <path> or --batch. Use --help for options.")
