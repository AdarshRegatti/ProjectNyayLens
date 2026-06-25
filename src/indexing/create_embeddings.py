"""
Generate embeddings in CHUNKS - GPU-safe with resume capability
Processes 20K paragraphs at a time with cooling breaks
"""

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import time

class ChunkedEmbeddingGenerator:
    """Generate embeddings in safe chunks with resume capability"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", chunk_size: int = 20000):
        """
        Initialize with sentence transformer
        chunk_size: Process this many paragraphs at a time
        """
        print(f"Loading model: {model_name}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available! GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            self.device = 'cuda'
        else:
            print("⚠️  CUDA not available, using CPU")
            self.device = 'cpu'
        
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.chunk_size = chunk_size
        print(f"✓ Model loaded on: {self.device}")
        print(f"✓ Chunk size: {chunk_size:,} paragraphs")
    
    def load_paragraphs(self, index_file: Path):
        """Load paragraphs from JSONL"""
        print("\nLoading paragraphs from index...")
        paragraphs = []
        
        with open(index_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading index"):
                para = json.loads(line)
                paragraphs.append(para)
        
        print(f"✓ Loaded {len(paragraphs):,} paragraphs")
        return paragraphs
    
    def process_chunk(self, texts, batch_size=64):
        """Process one chunk of texts"""
        embeddings_list = []
        
        with tqdm(total=len(texts), desc="Processing chunk", unit="para") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    device=self.device
                )
                
                embeddings_list.append(batch_embeddings)
                pbar.update(len(batch))
                
                # Clear GPU cache every 10 batches
                if self.device == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        return np.vstack(embeddings_list)
    
    def generate_embeddings_chunked(self, index_file: Path, output_dir: Path, batch_size: int = 64):
        """Generate embeddings in chunks with resume capability"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        # Load paragraphs
        paragraphs = self.load_paragraphs(index_file)
        total_paras = len(paragraphs)
        
        # Calculate chunks
        num_chunks = (total_paras + self.chunk_size - 1) // self.chunk_size
        print(f"\n✓ Will process in {num_chunks} chunks of {self.chunk_size:,} paragraphs each")
        
        # Check for existing chunks
        existing_chunks = list(chunks_dir.glob("chunk_*.npy"))
        completed_chunks = len(existing_chunks)
        
        if completed_chunks > 0:
            print(f"✓ Found {completed_chunks} existing chunks")
            response = input(f"Resume from chunk {completed_chunks + 1}? (yes/no): ")
            if response.lower() != 'yes':
                print("Starting fresh...")
                for f in existing_chunks:
                    f.unlink()
                completed_chunks = 0
        
        # Process chunks
        start_time = time.time()
        
        for chunk_idx in range(completed_chunks, num_chunks):
            print(f"\n{'='*70}")
            print(f"CHUNK {chunk_idx + 1}/{num_chunks}")
            print(f"{'='*70}")
            
            # Get chunk data
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_paras)
            chunk_paras = paragraphs[start_idx:end_idx]
            
            print(f"Processing paragraphs {start_idx:,} to {end_idx:,}")
            
            # Extract texts
            texts = [p['text'] for p in chunk_paras]
            ids = [p['id'] for p in chunk_paras]
            
            # Process chunk
            chunk_start = time.time()
            embeddings = self.process_chunk(texts, batch_size)
            chunk_time = time.time() - chunk_start
            
            print(f"✓ Chunk completed in {chunk_time/60:.1f} minutes")
            print(f"  Speed: {len(texts)/chunk_time:.1f} para/s")
            
            # Save chunk
            chunk_file = chunks_dir / f"chunk_{chunk_idx:03d}.npy"
            ids_file = chunks_dir / f"chunk_{chunk_idx:03d}_ids.json"
            
            np.save(chunk_file, embeddings)
            with open(ids_file, 'w') as f:
                json.dump(ids, f)
            
            print(f"✓ Saved to: {chunk_file.name}")
            
            # Clear GPU and add cooling break
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                if chunk_idx < num_chunks - 1:  # Not last chunk
                    print("\n⏸️  Cooling break: 10 seconds...")
                    time.sleep(10)
        
        # Combine all chunks
        print(f"\n{'='*70}")
        print("COMBINING CHUNKS...")
        print(f"{'='*70}")
        
        all_embeddings = []
        all_ids = []
        
        for chunk_idx in tqdm(range(num_chunks), desc="Loading chunks"):
            chunk_file = chunks_dir / f"chunk_{chunk_idx:03d}.npy"
            ids_file = chunks_dir / f"chunk_{chunk_idx:03d}_ids.json"
            
            embeddings = np.load(chunk_file)
            with open(ids_file, 'r') as f:
                ids = json.load(f)
            
            all_embeddings.append(embeddings)
            all_ids.extend(ids)
        
        final_embeddings = np.vstack(all_embeddings)
        
        print(f"✓ Combined shape: {final_embeddings.shape}")
        
        # Save final files
        print("\nSaving final files...")
        
        embeddings_file = output_dir / "paragraph_embeddings.npy"
        np.save(embeddings_file, final_embeddings)
        print(f"✓ Saved: {embeddings_file}")
        
        ids_file = output_dir / "paragraph_ids.json"
        with open(ids_file, 'w') as f:
            json.dump(all_ids, f)
        print(f"✓ Saved: {ids_file}")
        
        # Save metadata
        total_time = time.time() - start_time
        metadata = {
            'model_name': "BAAI/bge-base-en-v1.5",
            'embedding_dim': int(final_embeddings.shape[1]),
            'total_paragraphs': len(all_ids),
            'device_used': self.device,
            'batch_size': batch_size,
            'chunk_size': self.chunk_size,
            'num_chunks': num_chunks,
            'total_time_minutes': total_time / 60
        }
        
        with open(output_dir / "embedding_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Total time: {total_time/60:.1f} minutes")
        print(f"  Average speed: {len(all_ids)/total_time:.1f} para/s")
        
        return final_embeddings, all_ids


if __name__ == "__main__":
    print("="*70)
    print("NyayLens - Chunked Embedding Generation (GPU-Safe)")
    print("="*70)
    print()
    
    generator = ChunkedEmbeddingGenerator(
        model_name="BAAI/bge-base-en-v1.5",
        chunk_size=20000  # Process 20K at a time
    )
    
    embeddings, ids = generator.generate_embeddings_chunked(
        index_file=Path("data/processed/indexed/paragraph_index.jsonl"),
        output_dir=Path("data/processed/embeddings"),
        batch_size=64  # Reduced for safety
    )
    
    print()
    print("="*70)
    print(f"✓ COMPLETE! Generated {len(embeddings):,} embeddings")
    print("="*70)
