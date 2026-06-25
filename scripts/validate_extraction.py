# scripts/validate_extraction.py
"""Quick quality check on extracted data"""

import json
from pathlib import Path
from collections import Counter
import pandas as pd

def analyze_extraction():
    """Generate extraction quality report"""
    
    output_dir = Path("data/processed/extracted")
    
    # Load stats
    with open(output_dir / "processing_stats.json") as f:
        stats = json.load(f)
    
    print("="*70)
    print("EXTRACTION QUALITY REPORT")
    print("="*70)
    print(f"Total: {stats['total']:,}")
    print(f"Success: {stats['successful']:,} ({stats['success_rate']:.2f}%)")
    print(f"Failed: {stats['failed']:,}")
    print()
    
    # Analyze metadata
    metadata_files = list((output_dir / "metadata").glob("*.json"))
    print(f"Metadata files: {len(metadata_files):,}")
    
    methods = []
    quality_scores = []
    ocr_count = 0
    low_quality = []
    
    for meta_file in metadata_files:
        with open(meta_file) as f:
            meta = json.load(f)
            methods.append(meta['extraction_method'])
            quality_scores.append(meta['quality_score'])
            if meta['ocr_used']:
                ocr_count += 1
            if meta['quality_score'] < 0.5:
                low_quality.append((meta['filename'], meta['quality_score']))
    
    # Summary
    print("\nExtraction Methods:")
    for method, count in Counter(methods).most_common():
        print(f"  {method}: {count:,} ({count/len(methods)*100:.1f}%)")
    
    print(f"\nOCR Usage: {ocr_count:,} files ({ocr_count/len(metadata_files)*100:.1f}%)")
    
    avg_quality = sum(quality_scores) / len(quality_scores)
    print(f"\nAverage Quality Score: {avg_quality:.3f}")
    print(f"Low quality (<0.5): {len(low_quality)} files")
    
    if low_quality[:10]:
        print("\nSample low-quality files:")
        for fname, score in low_quality[:10]:
            print(f"  {fname}: {score:.3f}")
    
    # Year distribution
    year_counts = Counter()
    text_files = list((output_dir / "texts").glob("*.txt"))
    
    for txt_file in text_files:
        year = txt_file.stem.split('_')[0]
        year_counts[year] += 1
    
    print("\nYear Distribution (top 10):")
    for year, count in sorted(year_counts.items(), key=lambda x: x[0], reverse=True)[:10]:
        print(f"  {year}: {count:,}")
    
    print("\n" + "="*70)
    
    return {
        'total_extracted': len(metadata_files),
        'avg_quality': avg_quality,
        'ocr_count': ocr_count
    }

if __name__ == "__main__":
    analyze_extraction()
