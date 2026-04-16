"""
ingest.py - Parse Word doc transcripts and store in SQLite with OpenAI embeddings.
Drop .docx files into the /transcripts folder and run this script.
"""

import os, re, sys, json, sqlite3, time, hashlib
from docx import Document

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "db.sqlite")
TRANSCRIPTS_DIR = os.environ.get("TRANSCRIPTS_DIR", "transcripts")
CHUNK_SIZE = 400  # words per chunk


def init_db(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            filename    TEXT,
            file_hash   TEXT
        )
    """)
    # Add file_hash column to existing databases that predate this change
    try:
        cur.execute("ALTER TABLE episodes ADD COLUMN file_hash TEXT")
    except:
        pass
    cur.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id  TEXT NOT NULL,
            speaker     TEXT,
            timestamp   TEXT,
            start_secs  REAL,
            text        TEXT NOT NULL,
            embedding   TEXT,
            FOREIGN KEY (episode_id) REFERENCES episodes(id)
        )
    """)
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_segments
        USING fts5(episode_id, text, content='')
    """)
    conn.commit()


def extract_docx_text(filepath):
    """Extract text from docx, preserving bold markers as **text** for the parser."""
    doc = Document(filepath)
    lines = []
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        line = "".join(
            f"**{run.text}**" if run.bold else run.text
            for run in para.runs
        )
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def parse_timestamp(ts_str):
    """Convert [HH:MM:SS] or [MM:SS] to float seconds."""
    ts_str = ts_str.strip("[]")
    parts = ts_str.split(":")
    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0
    return 0.0


def parse_transcript(text):
    """
    Parse docx transcript into segments.
    Format: **Speaker N** [HH:MM:SS] text...
    Returns list of {speaker, timestamp, start_secs, text}
    """
    segments = []

    # Match lines like: **Speaker 2** [00:00:16] What is up guys...
    pattern = re.compile(
        r'\*\*([^*]+)\*\*\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*?)(?=\*\*[^*]+\*\*\s*\[|\Z)',
        re.DOTALL
    )

    for match in pattern.finditer(text):
        speaker   = match.group(1).strip()
        timestamp = match.group(2).strip()
        content   = match.group(3).strip().replace("\n", " ")
        content   = re.sub(r'\s+', ' ', content)

        if not content:
            continue

        segments.append({
            "speaker":    speaker,
            "timestamp":  timestamp,
            "start_secs": parse_timestamp(timestamp),
            "text":       content
        })

    return segments


def chunk_segments(segments, chunk_size=CHUNK_SIZE):
    """
    Merge short segments into chunks of ~chunk_size words.
    Preserves the timestamp and speaker of the first segment in each chunk.
    """
    chunks = []
    current_text  = ""
    current_ts    = ""
    current_secs  = 0.0
    current_spk   = ""
    word_count    = 0

    for seg in segments:
        words = seg["text"].split()
        if word_count + len(words) > chunk_size and current_text:
            chunks.append({
                "speaker":   current_spk,
                "timestamp": current_ts,
                "start_secs": current_secs,
                "text":      current_text.strip()
            })
            current_text  = ""
            word_count    = 0

        if not current_text:
            current_ts   = seg["timestamp"]
            current_secs = seg["start_secs"]
            current_spk  = seg["speaker"]

        current_text += " " + seg["text"]
        word_count   += len(words)

    if current_text.strip():
        chunks.append({
            "speaker":    current_spk,
            "timestamp":  current_ts,
            "start_secs": current_secs,
            "text":       current_text.strip()
        })

    return chunks


def get_embedding(text, retries=3):
    """Get OpenAI embedding for a text chunk."""
    if not OPENAI_API_KEY:
        return None

    import urllib.request, json as jsonlib

    payload = jsonlib.dumps({
        "model": "text-embedding-3-small",
        "input": text[:8000]
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = jsonlib.loads(resp.read())
                return data["data"][0]["embedding"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️  Embedding failed: {e}")
                return None


def compute_hash(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        md5.update(f.read())
    return md5.hexdigest()


def ingest_file(conn, filepath):
    """Ingest a single .docx transcript file. Re-ingests if the file has changed."""
    cur = conn.cursor()
    filename = os.path.basename(filepath)
    episode_id = filename.replace(".docx", "").replace(".DOCX", "")
    fhash = compute_hash(filepath)

    # Check if already ingested with same content
    existing = cur.execute(
        "SELECT id, file_hash FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if existing:
        if existing["file_hash"] == fhash:
            print(f"  ⏭️  Unchanged, skipping: {filename}")
            return 0
        # File changed — wipe and re-ingest
        print(f"  🔄  File changed, re-ingesting: {filename}")
        cur.execute("DELETE FROM segments WHERE episode_id = ?", (episode_id,))
        cur.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        conn.commit()

    # Extract text
    raw_text = extract_docx_text(filepath)
    if not raw_text.strip():
        print(f"  ⚠️  Could not extract text from: {filename}")
        return 0

    # Get title from first bold line
    title_match = re.search(r'\*\*([^*\n]+)\*\*', raw_text)
    title = title_match.group(1).strip() if title_match else episode_id

    # Parse into segments
    segments = parse_transcript(raw_text)
    if not segments:
        print(f"  ⚠️  No segments parsed from: {filename}")
        return 0

    # Chunk segments
    chunks = chunk_segments(segments)

    # Store episode
    cur.execute(
        "INSERT OR REPLACE INTO episodes (id, title, filename, file_hash) VALUES (?, ?, ?, ?)",
        (episode_id, title, filename, fhash)
    )

    # Store chunks with embeddings
    stored = 0
    for i, chunk in enumerate(chunks):
        # Get embedding if API key available
        embedding = None
        if OPENAI_API_KEY:
            embedding = get_embedding(chunk["text"])
            if embedding:
                embedding = json.dumps(embedding)

        cur.execute("""
            INSERT INTO segments (episode_id, speaker, timestamp, start_secs, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (episode_id, chunk["speaker"], chunk["timestamp"],
              chunk["start_secs"], chunk["text"], embedding))

        cur.execute(
            "INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)",
            (episode_id, chunk["text"])
        )
        stored += 1

        if OPENAI_API_KEY and (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(chunks)} chunks embedded")

    conn.commit()
    return stored


def main():
    if not os.path.exists(TRANSCRIPTS_DIR):
        os.makedirs(TRANSCRIPTS_DIR)
        print(f"📁 Created '{TRANSCRIPTS_DIR}' folder. Drop your .docx files in there and re-run.")
        sys.exit(0)

    docx_files = [
        f for f in os.listdir(TRANSCRIPTS_DIR)
        if f.lower().endswith(".docx")
    ]

    if not docx_files:
        print(f"No .docx files found in '{TRANSCRIPTS_DIR}'. Add some and re-run.")
        sys.exit(0)

    if not OPENAI_API_KEY:
        print("⚠️  No OPENAI_API_KEY set — keyword search only, no semantic search.")
        print("    Set it for AI-powered semantic search.\n")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    print(f"Found {len(docx_files)} transcript(s) to ingest\n")

    total = 0
    reingest_happened = False
    for i, fname in enumerate(sorted(docx_files), 1):
        fpath = os.path.join(TRANSCRIPTS_DIR, fname)
        print(f"[{i}/{len(docx_files)}] {fname}")
        count = ingest_file(conn, fpath)
        print(f"    → {count} chunks stored")
        if count > 0:
            reingest_happened = True
        total += count

    # If any episodes were (re-)ingested, rebuild FTS from scratch so
    # stale entries from old versions don't pollute keyword search.
    if reingest_happened:
        print("\nRebuilding FTS index...")
        cur = conn.cursor()
        cur.execute("INSERT INTO fts_segments(fts_segments) VALUES('delete-all')")
        rows = conn.execute("SELECT episode_id, text FROM segments").fetchall()
        for row in rows:
            cur.execute("INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)", row)
        conn.commit()
        print(f"FTS rebuilt with {len(rows)} entries.")

    conn.close()
    print(f"\n✅ Done. {total} total chunks stored in {DB_PATH}")


if __name__ == "__main__":
    main()
