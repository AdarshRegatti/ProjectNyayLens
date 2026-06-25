# """
# Batch processor for large-scale PDF extraction
# Includes progress tracking, error handling, and resumability
# """

# from pathlib import Path
# from typing import List, Dict
# import logging
# from tqdm import tqdm
# import json
# from datetime import datetime
# from pdf_extractor import LegalJudgmentExtractor

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class BatchProcessor:
#     """
#     Batch processor for extracting text from thousands of legal judgments
#     Features: Progress tracking, resumability, comprehensive reporting
#     """
    
#     def __init__(self, 
#                  input_dir: Path, 
#                  output_dir: Path,
#                  num_workers: int = 1,
#                  enable_ocr: bool = True):
        
#         self.input_dir = Path(input_dir)
#         self.output_dir = Path(output_dir)
#         self.num_workers = num_workers
        
#         # Create extractor
#         self.extractor = LegalJudgmentExtractor(output_dir, enable_ocr=enable_ocr)
        
#         # Progress tracking
#         self.progress_file = output_dir / "processing_progress.json"
#         self.stats_file = output_dir / "processing_stats.json"
        
#     def get_all_pdfs(self) -> List[Path]:
#         """Get all PDF files from input directory"""
#         return sorted(self.input_dir.rglob("*.pdf"))
    
#     def load_progress(self) -> set:
#         """Load already processed files"""
#         if self.progress_file.exists():
#             with open(self.progress_file, 'r') as f:
#                 data = json.load(f)
#                 return set(data.get('processed_files', []))
#         return set()
    
#     def save_progress(self, processed_files: set):
#         """Save processing progress"""
#         with open(self.progress_file, 'w') as f:
#             json.dump({
#                 'processed_files': list(processed_files),
#                 'last_updated': datetime.now().isoformat(),
#                 'total_processed': len(processed_files)
#             }, f, indent=2)
    
#     def process_single_pdf(self, pdf_path: Path) -> Dict:
#         """Process a single PDF"""
#         try:
#             success = self.extractor.process_pdf(pdf_path)
#             return {
#                 'filename': pdf_path.name,
#                 'year': pdf_path.parent.name,
#                 'success': success,
#                 'error': None
#             }
#         except Exception as e:
#             return {
#                 'filename': pdf_path.name,
#                 'year': pdf_path.parent.name,
#                 'success': False,
#                 'error': str(e)
#             }
    
#     def process_batch(self, 
#                      start_year: int = None, 
#                      end_year: int = None,
#                      limit: int = None,
#                      resume: bool = True):
#         """
#         Process PDFs in batch with progress tracking
        
#         Args:
#             start_year: Start from this year (inclusive)
#             end_year: Process until this year (inclusive)
#             limit: Maximum number of PDFs to process
#             resume: Continue from last checkpoint
#         """
        
#         logger.info("Starting batch processing...")
#         logger.info(f"Workers: {self.num_workers}")
        
#         # Get all PDFs
#         all_pdfs = self.get_all_pdfs()
#         logger.info(f"Found {len(all_pdfs):,} PDFs")
        
#         # Filter by year if specified
#         if start_year or end_year:
#             all_pdfs = [
#                 p for p in all_pdfs 
#                 if (not start_year or int(p.parent.name) >= start_year) and
#                    (not end_year or int(p.parent.name) <= end_year)
#             ]
#             logger.info(f"Filtered to {len(all_pdfs):,} PDFs (years {start_year}-{end_year})")
        
#         # Load progress and filter already processed
#         if resume:
#             processed = self.load_progress()
#             all_pdfs = [p for p in all_pdfs if str(p) not in processed]
#             logger.info(f"Resuming: {len(all_pdfs):,} PDFs remaining")
#         else:
#             processed = set()
        
#         # Apply limit
#         if limit:
#             all_pdfs = all_pdfs[:limit]
#             logger.info(f"Limited to {len(all_pdfs):,} PDFs")
        
#         if not all_pdfs:
#             logger.info("No PDFs to process!")
#             return
        
#         # Initialize stats
#         stats = {
#             'total': len(all_pdfs),
#             'successful': 0,
#             'failed': 0,
#             'start_time': datetime.now().isoformat(),
#             'failed_files': []
#         }
        
#         # Process with progress bar
#         with tqdm(total=len(all_pdfs), desc="Processing PDFs") as pbar:
#             for pdf_path in all_pdfs:
#                 result = self.process_single_pdf(pdf_path)
                
#                 if result['success']:
#                     stats['successful'] += 1
#                 else:
#                     stats['failed'] += 1
#                     stats['failed_files'].append({
#                         'file': result['filename'],
#                         'error': result['error']
#                     })
#                     logger.warning(f"Failed: {result['filename']} - {result['error']}")
                
#                 # Update progress
#                 processed.add(str(pdf_path))
                
#                 # Save progress every 50 files
#                 if len(processed) % 50 == 0:
#                     self.save_progress(processed)
                
#                 pbar.update(1)
#                 pbar.set_postfix({
#                     'Success': stats['successful'],
#                     'Failed': stats['failed']
#                 })
        
#         # Final save
#         self.save_progress(processed)
        
#         # Save statistics
#         stats['end_time'] = datetime.now().isoformat()
#         stats['success_rate'] = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
#         with open(self.stats_file, 'w') as f:
#             json.dump(stats, f, indent=2)
        
#         # Summary
#         logger.info("\n" + "="*60)
#         logger.info("PROCESSING COMPLETE")
#         logger.info("="*60)
#         logger.info(f"Total processed: {stats['total']:,}")
#         logger.info(f"Successful: {stats['successful']:,}")
#         logger.info(f"Failed: {stats['failed']:,}")
#         logger.info(f"Success rate: {stats['success_rate']:.2f}%")
#         logger.info("="*60)


# def main():
#     """Main execution with CLI arguments"""
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Batch process legal judgment PDFs')
#     parser.add_argument('--start-year', type=int, help='Start year (inclusive)')
#     parser.add_argument('--end-year', type=int, help='End year (inclusive)')
#     parser.add_argument('--limit', type=int, help='Maximum PDFs to process')
#     parser.add_argument('--no-resume', action='store_true', help='Start fresh')
#     parser.add_argument('--no-ocr', action='store_true', help='Disable OCR fallback')
    
#     args = parser.parse_args()
    
#     # Configuration
#     INPUT_DIR = Path("data/raw")
#     OUTPUT_DIR = Path("data/processed/extracted")
    
#     processor = BatchProcessor(
#         input_dir=INPUT_DIR,
#         output_dir=OUTPUT_DIR,
#         enable_ocr=not args.no_ocr
#     )
    
#     processor.process_batch(
#         start_year=args.start_year,
#         end_year=args.end_year,
#         limit=args.limit,
#         resume=not args.no_resume
#     )


# if __name__ == "__main__":
#     main()
"""
Batch processor for large-scale PDF extraction
Includes progress tracking, error handling, and resumability
"""

from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import json
from datetime import datetime
from pdf_extractor import LegalJudgmentExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for extracting text from thousands of legal judgments
    Features: Progress tracking, resumability, comprehensive reporting
    """
    
    def __init__(self, 
                 input_dir: Path, 
                 output_dir: Path,
                 num_workers: int = 1,
                 enable_ocr: bool = True):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        
        # Create extractor
        self.extractor = LegalJudgmentExtractor(output_dir, enable_ocr=enable_ocr)
        
        # Progress tracking
        self.progress_file = output_dir / "processing_progress.json"
        self.stats_file = output_dir / "processing_stats.json"
        
    def get_all_pdfs(self) -> List[Path]:
        """Get all PDF files (case-insensitive)"""
        # FIX: Search for both .pdf and .PDF
        pdfs_lower = list(self.input_dir.rglob("*.pdf"))
        pdfs_upper = list(self.input_dir.rglob("*.PDF"))
        all_pdfs = pdfs_lower + pdfs_upper
        return sorted(set(all_pdfs))  # Remove duplicates and sort
    
    def load_progress(self) -> set:
        """Load already processed files"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        return set()
    
    def save_progress(self, processed_files: set):
        """Save processing progress"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'processed_files': list(processed_files),
                'last_updated': datetime.now().isoformat(),
                'total_processed': len(processed_files)
            }, f, indent=2)
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF"""
        try:
            success = self.extractor.process_pdf(pdf_path)
            return {
                'filename': pdf_path.name,
                'year': pdf_path.parent.name,
                'success': success,
                'error': None
            }
        except Exception as e:
            return {
                'filename': pdf_path.name,
                'year': pdf_path.parent.name,
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, 
                     start_year: int = None, 
                     end_year: int = None,
                     limit: int = None,
                     resume: bool = True):
        """
        Process PDFs in batch with progress tracking
        
        Args:
            start_year: Start from this year (inclusive)
            end_year: Process until this year (inclusive)
            limit: Maximum number of PDFs to process
            resume: Continue from last checkpoint
        """
        
        logger.info("Starting batch processing...")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Get all PDFs
        all_pdfs = self.get_all_pdfs()
        logger.info(f"Found {len(all_pdfs):,} PDFs")
        
        if len(all_pdfs) == 0:
            logger.error("❌ No PDFs found! Check your data/raw directory.")
            logger.error(f"Looking in: {self.input_dir}")
            logger.error("Make sure PDFs are in year folders like: data/raw/1950/*.PDF")
            return
        
        # Filter by year if specified
        if start_year or end_year:
            filtered_pdfs = []
            for p in all_pdfs:
                try:
                    year = int(p.parent.name)
                    if (not start_year or year >= start_year) and (not end_year or year <= end_year):
                        filtered_pdfs.append(p)
                except ValueError:
                    logger.warning(f"Skipping non-year folder: {p.parent.name}")
            
            all_pdfs = filtered_pdfs
            logger.info(f"Filtered to {len(all_pdfs):,} PDFs (years {start_year}-{end_year})")
        
        # Load progress and filter already processed
        if resume:
            processed = self.load_progress()
            all_pdfs = [p for p in all_pdfs if str(p) not in processed]
            logger.info(f"Resuming: {len(all_pdfs):,} PDFs remaining")
        else:
            processed = set()
        
        # Apply limit
        if limit:
            all_pdfs = all_pdfs[:limit]
            logger.info(f"Limited to {len(all_pdfs):,} PDFs")
        
        if not all_pdfs:
            logger.info("No PDFs to process!")
            return
        
        # Initialize stats
        stats = {
            'total': len(all_pdfs),
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now().isoformat(),
            'failed_files': []
        }
        
        # Process with progress bar
        with tqdm(total=len(all_pdfs), desc="Processing PDFs") as pbar:
            for pdf_path in all_pdfs:
                result = self.process_single_pdf(pdf_path)
                
                if result['success']:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_files'].append({
                        'file': result['filename'],
                        'year': result['year'],
                        'error': result['error']
                    })
                    logger.warning(f"Failed: {result['filename']} - {result['error']}")
                
                # Update progress
                processed.add(str(pdf_path))
                
                # Save progress every 50 files
                if len(processed) % 50 == 0:
                    self.save_progress(processed)
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': stats['successful'],
                    'Failed': stats['failed']
                })
        
        # Final save
        self.save_progress(processed)
        
        # Save statistics
        stats['end_time'] = datetime.now().isoformat()
        stats['success_rate'] = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total processed: {stats['total']:,}")
        logger.info(f"Successful: {stats['successful']:,}")
        logger.info(f"Failed: {stats['failed']:,}")
        logger.info(f"Success rate: {stats['success_rate']:.2f}%")
        logger.info("="*60)


def main():
    """Main execution with CLI arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process legal judgment PDFs')
    parser.add_argument('--start-year', type=int, help='Start year (inclusive)')
    parser.add_argument('--end-year', type=int, help='End year (inclusive)')
    parser.add_argument('--limit', type=int, help='Maximum PDFs to process')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    parser.add_argument('--no-ocr', action='store_true', help='Disable OCR fallback')
    
    args = parser.parse_args()
    
    # Configuration
    INPUT_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed/extracted")
    
    processor = BatchProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        enable_ocr=not args.no_ocr
    )
    
    processor.process_batch(
        start_year=args.start_year,
        end_year=args.end_year,
        limit=args.limit,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
