# """Create SQLite index for fast paragraph lookup"""

# import sqlite3
# import json
# from pathlib import Path
# from tqdm import tqdm

# def create_sqlite_index():
#     print("Creating SQLite index...")
    
#     db_path = Path("data/processed/indexed/paragraphs.db")
#     db_path.parent.mkdir(parents=True, exist_ok=True)
    
#     # Create database
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()
    
#     # Create table
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS paragraphs (
#             id TEXT PRIMARY KEY,
#             judgment_id TEXT,
#             page_no INTEGER,
#             text TEXT,
#             char_count INTEGER,
#             word_count INTEGER
#         )
#     """)
    
#     cursor.execute("CREATE INDEX IF NOT EXISTS idx_judgment ON paragraphs(judgment_id)")
    
#     # Load data
#     index_file = Path("data/processed/indexed/paragraph_index.jsonl")
    
#     with open(index_file, 'r', encoding='utf-8') as f:
#         total = sum(1 for _ in f)
    
#     with open(index_file, 'r', encoding='utf-8') as f:
#         batch = []
#         for line in tqdm(f, total=total, desc="Inserting"):
#             p = json.loads(line)
#             batch.append((
#                 p['id'], p['judgment_id'], p['page_no'],
#                 p['text'], p['char_count'], p['word_count']
#             ))
            
#             if len(batch) >= 1000:
#                 cursor.executemany(
#                     "INSERT OR REPLACE INTO paragraphs VALUES (?,?,?,?,?,?)",
#                     batch
#                 )
#                 batch = []
        
#         if batch:
#             cursor.executemany(
#                 "INSERT OR REPLACE INTO paragraphs VALUES (?,?,?,?,?,?)",
#                 batch
#             )
    
#     conn.commit()
#     conn.close()
    
#     print(f"✓ SQLite index created: {db_path}")

# if __name__ == "__main__":
#     create_sqlite_index()
"""
Create SQLite index with section annotations
Source: paragraph_index_with_sections.jsonl
"""

import sqlite3
import json
from pathlib import Path
from tqdm import tqdm


INPUT_INDEX = Path("data/processed/indexed/paragraph_index_with_sections.jsonl")
DB_PATH = Path("data/processed/indexed/paragraphs.db")


def create_sqlite_index():
    print("=" * 70)
    print("NyayLens – Creating SQLite Index (with Sections)")
    print("=" * 70)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop existing table (derived data → safe to rebuild)
    cursor.execute("DROP TABLE IF EXISTS paragraphs")

    # Create table
    cursor.execute("""
        CREATE TABLE paragraphs (
            id TEXT PRIMARY KEY,
            judgment_id TEXT,
            page_no INTEGER,
            text TEXT,
            char_count INTEGER,
            word_count INTEGER,
            section TEXT,
            section_conf REAL
        )
    """)

    # Create FTS5 virtual table for fast full-text search (BM25)
    cursor.execute("DROP TABLE IF EXISTS paragraphs_fts")
    cursor.execute("""
        CREATE VIRTUAL TABLE paragraphs_fts USING fts5(
            id UNINDEXED,
            text,
            tokenize='porter unicode61'
        )
    """)

    # Indexes for fast lookup
    cursor.execute("CREATE INDEX idx_judgment_id ON paragraphs(judgment_id)")
    cursor.execute("CREATE INDEX idx_section ON paragraphs(section)")
    cursor.execute("CREATE INDEX idx_judgment_section ON paragraphs(judgment_id, section)")

    conn.commit()

    # Count total records
    with open(INPUT_INDEX, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    print(f"✓ Inserting {total:,} paragraphs")

    # Insert data in batches
    batch = []
    BATCH_SIZE = 1000

    with open(INPUT_INDEX, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Inserting"):
            p = json.loads(line)

            batch.append((
                p["id"],
                p["judgment_id"],
                p.get("page_no", -1),
                p["text"],
                p.get("char_count", len(p["text"])),
                p.get("word_count", len(p["text"].split())),
                p.get("section", "unknown"),
                p.get("section_conf", 0.0),
            ))

            if len(batch) >= BATCH_SIZE:
                cursor.executemany(
                    """
                    INSERT INTO paragraphs
                    (id, judgment_id, page_no, text, char_count, word_count, section, section_conf)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch
                )
                
                # Insert into FTS5 table
                fts_batch = [(b[0], b[3]) for b in batch]
                cursor.executemany(
                    "INSERT INTO paragraphs_fts (id, text) VALUES (?, ?)",
                    fts_batch
                )
                
                batch.clear()

        if batch:
            cursor.executemany(
                """
                INSERT INTO paragraphs
                (id, judgment_id, page_no, text, char_count, word_count, section, section_conf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch
            )
            
            fts_batch = [(b[0], b[3]) for b in batch]
            cursor.executemany(
                "INSERT INTO paragraphs_fts (id, text) VALUES (?, ?)",
                fts_batch
            )

    conn.commit()
    conn.close()

    print("\n✓ SQLite index created successfully")
    print(f"✓ Database path: {DB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    create_sqlite_index()
