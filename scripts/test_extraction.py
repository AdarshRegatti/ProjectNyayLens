"""
Test extraction pipeline on sample data
"""

from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "extraction"))

# Change to project root
os.chdir(Path(__file__).parent.parent)

from batch_processor import BatchProcessor

def main():
    print("="*70)
    print(" "*15 + "NyayLens - PDF Extraction Test")
    print("="*70)
    print()
    
    # Verify paths exist
    INPUT_DIR = Path("data/raw")
    OUTPUT_DIR = Path("data/processed/extracted")
    
    if not INPUT_DIR.exists():
        print(f"❌ ERROR: Input directory not found: {INPUT_DIR}")
        print(f"Current directory: {Path.cwd()}")
        return
    
    print(f"✓ Input directory found: {INPUT_DIR}")
    print()
    
    processor = BatchProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_workers=1,
        enable_ocr=False
    )
    
    print("📋 Test Plan:")
    print("  - Process: 2024-2025 judgments")
    print("  - Limit: 100 PDFs")
    print("  - OCR: Disabled (for speed)")
    print("  - Resume: Disabled (fresh start)")
    print()
    
    input("Press Enter to start extraction...")
    print()
    
    # Run extraction
    processor.process_batch(
        start_year=2024,
        end_year=2025,
        limit=100,
        resume=False
    )
    
    print()
    print("="*70)
    print("✓ Test complete!")
    print()
    print("📁 Check outputs:")
    print(f"  - Texts: {OUTPUT_DIR}/texts/")
    print(f"  - Metadata: {OUTPUT_DIR}/metadata/")
    print(f"  - Stats: {OUTPUT_DIR}/processing_stats.json")
    print("="*70)

if __name__ == "__main__":
    main()
