"""
Production PDF Extractor for Legal Judgments
Enhanced with robust error handling, quality checks, and paragraph preservation
"""

import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import re

# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR libraries not installed. OCR fallback disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionMetadata:
    """Metadata for extracted judgment"""
    filename: str
    year: str
    num_pages: int
    text_length: int
    extraction_method: str
    has_text: bool
    extraction_timestamp: str
    file_size_bytes: int
    ocr_used: bool
    quality_score: float
    paragraph_count: int
    errors: List[str]
    warnings: List[str]


class TextQualityChecker:
    """Utility class for assessing extracted text quality"""
    
    # Legal keywords to preserve even in short lines
    LEGAL_KEYWORDS = {
        'held', 'order', 'appeal', 'writ', 'judgment', 'decree',
        'petition', 'application', 'allowed', 'dismissed', 'granted',
        'rejected', 'reserved', 'disposed', 'quashed', 'set aside',
        'affirmed', 'reversed', 'remanded', 'suo moto', 'ex parte',
        'interim', 'stay', 'injunction', 'bail', 'custody', 'liberty',
        'notice', 'respondent', 'petitioner', 'appellant', 'accused'
    }
    
    @staticmethod
    def calculate_quality_score(text: str) -> Tuple[float, List[str]]:
        """
        Calculate quality score (0-1) for extracted text
        
        Returns:
            (score, issues_found)
        """
        if not text or len(text.strip()) < 100:
            return 0.0, ["Text too short"]
        
        issues = []
        score = 1.0
        
        # Check 1: Alphabetic character ratio
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text.replace('\n', '').replace(' ', ''))
        
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            if alpha_ratio < 0.5:
                score -= 0.3
                issues.append(f"Low alphabetic ratio: {alpha_ratio:.2f}")
        
        # Check 2: Average word length (gibberish detection)
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 2 or avg_word_len > 15:
                score -= 0.2
                issues.append(f"Unusual avg word length: {avg_word_len:.1f}")
        
        # Check 3: Check for repeated patterns (OCR errors)
        lines = text.split('\n')
        if len(lines) > 10:
            unique_lines = len(set(line.strip() for line in lines if line.strip()))
            repetition_ratio = unique_lines / len(lines)
            if repetition_ratio < 0.3:
                score -= 0.2
                issues.append(f"High repetition: {repetition_ratio:.2f}")
        
        # Check 4: Minimum sentence structure
        sentence_markers = text.count('.') + text.count('?') + text.count('!')
        if len(words) > 100 and sentence_markers < len(words) / 50:
            score -= 0.1
            issues.append("Lacks sentence structure")
        
        return max(0.0, min(1.0, score)), issues
    
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """
        Normalize OCR-extracted text with legal-aware filtering
        - Remove excessive whitespace
        - Collapse multiple newlines
        - Remove repeated headers
        - Preserve important legal terms
        """
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Collapse multiple newlines (keep max 2 for paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        
        # Legal-aware line filtering
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Skip pure numbers (page numbers)
            if stripped.isdigit():
                continue
            
            # PRESERVE if:
            # 1. Line is substantial (>10 chars)
            # 2. Contains legal keyword (even if short like "Held.")
            # 3. Is alphabetic and reasonable length (>3 chars)
            if (len(stripped) > 10 or
                any(keyword in stripped.lower() for keyword in TextQualityChecker.LEGAL_KEYWORDS) or
                (stripped.replace('.', '').replace(',', '').isalpha() and len(stripped) > 3)):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove repeated header patterns
        lines = text.split('\n')
        result = []
        prev_line = None
        repeat_count = 0
        
        for line in lines:
            if line.strip() == prev_line and prev_line:
                repeat_count += 1
                if repeat_count < 2:  # Allow max 2 repetitions
                    result.append(line)
            else:
                repeat_count = 0
                result.append(line)
                prev_line = line.strip()
        
        return '\n'.join(result).strip()


class LegalJudgmentExtractor:
    """
    Production-grade extractor with robust error handling and quality assurance
    """
    
    def __init__(self, output_dir: Path, enable_ocr: bool = True, ocr_max_pages: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_max_pages = ocr_max_pages
        
        if enable_ocr and not OCR_AVAILABLE:
            logger.warning("OCR requested but libraries not installed.")
        
        # Create subdirectories
        self.text_dir = self.output_dir / "texts"
        self.metadata_dir = self.output_dir / "metadata"
        self.failed_dir = self.output_dir / "failed"
        self.ocr_log_file = self.output_dir / "ocr_cases.jsonl"
        
        for dir_path in [self.text_dir, self.metadata_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def extract_year_from_path(self, pdf_path: Path) -> Tuple[str, List[str]]:
        """
        Safely extract year from path with validation
        
        Returns:
            (year, warnings)
        """
        warnings = []
        year = pdf_path.parent.name
        
        # Validate year
        if not year.isdigit():
            warnings.append(f"Invalid year from directory: {year}")
            
            # Try to extract from filename
            filename = pdf_path.stem
            year_match = re.search(r'(19|20)\d{2}', filename)
            if year_match:
                year = year_match.group(0)
                warnings.append(f"Year extracted from filename: {year}")
            else:
                year = "unknown"
                warnings.append("Could not determine year")
        else:
            # Validate year range
            year_int = int(year)
            if year_int < 1950 or year_int > 2025:
                warnings.append(f"Year {year} outside expected range (1950-2025)")
        
        return year, warnings
    
    def count_paragraphs(self, text: str) -> int:
        """Count paragraph-like structures in text"""
        # Split by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        # Filter out very short "paragraphs" (likely headers)
        substantial_paragraphs = [p for p in paragraphs if len(p) > 50]
        return len(substantial_paragraphs)
    
    def extract_with_pypdf2(self, pdf_path: Path) -> Optional[str]:
        """Primary extraction - preserves paragraph structure"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text.strip())
                
                # Join with double newline to preserve page breaks
                full_text = "\n\n".join(text_parts)
                
                # Quality check
                score, _ = TextQualityChecker.calculate_quality_score(full_text)
                return full_text if score > 0.3 else None
                
        except Exception as e:
            logger.debug(f"PyPDF2 failed for {pdf_path.name}: {e}")
            return None
    
    def extract_with_pdfplumber(self, pdf_path: Path) -> Optional[str]:
        """Fallback extraction - better for complex layouts"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text.strip())
                
                full_text = "\n\n".join(text_parts)
                
                score, _ = TextQualityChecker.calculate_quality_score(full_text)
                return full_text if score > 0.3 else None
                
        except Exception as e:
            logger.debug(f"pdfplumber failed for {pdf_path.name}: {e}")
            return None
    
    def extract_with_ocr(self, pdf_path: Path, num_pages: int) -> Optional[str]:
        """
        OCR extraction with proper page limiting and text normalization
        
        Args:
            pdf_path: Path to PDF
            num_pages: Total pages in PDF (for proper limiting)
        """
        if not self.enable_ocr:
            return None
        
        try:
            logger.info(f"OCR extraction: {pdf_path.name}")
            
            # Proper page limiting
            last_page = min(self.ocr_max_pages, num_pages)
            
            if num_pages > self.ocr_max_pages:
                logger.warning(f"PDF has {num_pages} pages, OCR limited to first {self.ocr_max_pages}")
            
            # Convert to images
            images = convert_from_path(
                pdf_path,
                dpi=300,
                first_page=1,
                last_page=last_page
            )
            
            text_parts = []
            for i, image in enumerate(images, 1):
                logger.debug(f"OCR page {i}/{len(images)}")
                
                text = pytesseract.image_to_string(image, lang='eng')
                if text.strip():
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            # Normalize OCR text
            full_text = TextQualityChecker.clean_ocr_text(full_text)
            
            # Check quality
            score, issues = TextQualityChecker.calculate_quality_score(full_text)
            
            if score > 0.3:
                # Log successful OCR to JSONL
                self._log_ocr_case(pdf_path, num_pages, last_page, score)
                logger.info(f"✓ OCR successful (quality: {score:.2f})")
                return full_text
            else:
                logger.warning(f"OCR quality too low ({score:.2f}): {issues}")
                return None
                
        except Exception as e:
            logger.warning(f"OCR failed for {pdf_path.name}: {e}")
            return None
    
    def _log_ocr_case(self, pdf_path: Path, total_pages: int, pages_processed: int, quality: float):
        """Log OCR usage to JSONL file"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': pdf_path.name,
            'year': pdf_path.parent.name,
            'total_pages': total_pages,
            'pages_processed': pages_processed,
            'quality_score': quality
        }
        
        with open(self.ocr_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def extract_pdf(self, pdf_path: Path) -> Dict:
        """
        Main extraction with fallback chain and quality assurance
        """
        errors = []
        warnings = []
        text = None
        method = None
        ocr_used = False
        quality_score = 0.0
        
        # Get metadata
        file_size = pdf_path.stat().st_size
        
        # Robust year extraction
        year, year_warnings = self.extract_year_from_path(pdf_path)
        warnings.extend(year_warnings)
        
        # Count pages first (needed for OCR)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
        except Exception as e:
            num_pages = 0
            errors.append(f"Could not count pages: {e}")
        
        # Extraction chain: PyPDF2 → pdfplumber → OCR
        text = self.extract_with_pypdf2(pdf_path)
        if text:
            method = "pypdf2"
        else:
            errors.append("PyPDF2 insufficient")
            
            text = self.extract_with_pdfplumber(pdf_path)
            if text:
                method = "pdfplumber"
            else:
                errors.append("pdfplumber failed")
                
                if self.enable_ocr and num_pages > 0:
                    text = self.extract_with_ocr(pdf_path, num_pages)
                    if text:
                        method = "ocr"
                        ocr_used = True
                        warnings.append("OCR used - verify quality")
                    else:
                        errors.append("OCR failed")
        
        # Calculate quality
        paragraph_count = 0
        if text:
            quality_score, quality_issues = TextQualityChecker.calculate_quality_score(text)
            paragraph_count = self.count_paragraphs(text)
            
            if quality_score < 0.7:
                warnings.extend(quality_issues)
        
        # Create metadata
        metadata = ExtractionMetadata(
            filename=pdf_path.name,
            year=year,
            num_pages=num_pages,
            text_length=len(text) if text else 0,
            extraction_method=method if method else "failed",
            has_text=text is not None,
            extraction_timestamp=datetime.now().isoformat(),
            file_size_bytes=file_size,
            ocr_used=ocr_used,
            quality_score=quality_score,
            paragraph_count=paragraph_count,
            errors=errors,
            warnings=warnings
        )
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    def save_extraction(self, pdf_path: Path, extraction_result: Dict) -> bool:
        """Save with quality indicators"""
        
        metadata = extraction_result['metadata']
        text = extraction_result['text']
        
        base_name = pdf_path.stem
        year = metadata.year
        
        # Save text
        if text:
            text_file = self.text_dir / f"{year}_{base_name}.txt"
            try:
                with open(text_file, 'w', encoding='utf-8') as f:
                    # Add quality header
                    f.write(f"{'='*70}\n")
                    f.write(f"File: {metadata.filename}\n")
                    f.write(f"Extraction: {metadata.extraction_method}\n")
                    f.write(f"Quality: {metadata.quality_score:.2f}\n")
                    f.write(f"Paragraphs: {metadata.paragraph_count}\n")
                    
                    if metadata.ocr_used:
                        f.write("⚠️ OCR USED - Verify important details\n")
                    
                    if metadata.warnings:
                        f.write(f"Warnings: {', '.join(metadata.warnings[:3])}\n")
                    
                    f.write(f"{'='*70}\n\n")
                    f.write(text)
                    
            except Exception as e:
                logger.error(f"Failed to save text: {e}")
                return False
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{year}_{base_name}.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
        
        # Log failures
        if not text:
            failed_log = self.failed_dir / "failed_extractions.jsonl"
            with open(failed_log, 'a', encoding='utf-8') as f:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'file': str(pdf_path),
                    'errors': metadata.errors
                }
                f.write(json.dumps(log_entry) + '\n')
        
        return True
    
    def process_pdf(self, pdf_path: Path) -> bool:
        """Process single PDF"""
        try:
            result = self.extract_pdf(pdf_path)
            return self.save_extraction(pdf_path, result)
        except Exception as e:
            logger.error(f"Unexpected error: {pdf_path.name}: {e}")
            return False


if __name__ == "__main__":
    # Test
    print("="*70)
    print("Testing Enhanced PDF Extractor")
    print("="*70)
    
    extractor = LegalJudgmentExtractor(
        output_dir=Path("data/processed/extracted"),
        enable_ocr=False
    )
    
    test_pdf = Path("data/raw/2025/A_John_Kennedy_vs_The_State_Of_Tamil_Nadu_on_24_March_2025_1.PDF")
    
    if test_pdf.exists():
        print(f"\nTesting: {test_pdf.name}")
        success = extractor.process_pdf(test_pdf)
        print(f"\n{'✓' if success else '✗'} Extraction {'successful' if success else 'failed'}")
        
        # Show metadata
        metadata_file = Path("data/processed/extracted/metadata") / f"2025_{test_pdf.stem}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                print(f"\nMethod: {metadata['extraction_method']}")
                print(f"Quality: {metadata['quality_score']:.2f}")
                print(f"Paragraphs: {metadata['paragraph_count']}")
                print(f"Text length: {metadata['text_length']:,} chars")
    else:
        print("Test PDF not found")
