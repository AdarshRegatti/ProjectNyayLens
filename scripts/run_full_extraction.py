"""
Full extraction of all 26,688 PDFs
Run this after successful testing
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "extraction"))

from batch_processor import BatchProcessor

def main():
    print("="*70)
    print(" "*10 + "NyayLens - Full Dataset Extraction")
    print("="*70)
    print()
    print("⚠️  WARNING: This will process all 26,688 PDFs")
    print("   Estimated time: 7-15 hours")
    print("   Make sure you have:")
    print("     - Tested on sample data first")
    print("     - ~10GB free disk space")
    print("     - Stable power supply")
    print()
    
    response = input("Proceed? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    print()
    print("Starting full extraction...")
    print()
    
    processor = BatchProcessor(
        input_dir=Path("data/raw"),
        output_dir=Path("data/processed/extracted"),
        enable_ocr=True  # Enable OCR for full extraction
    )
    
    processor.process_batch(
        start_year=None,  # All years
        end_year=None,
        limit=None,  # No limit
        resume=True  # Resume if interrupted
    )
    
    print()
    print("="*70)
    print("✓ Full extraction complete!")
    print("="*70)

if __name__ == "__main__":
    main()