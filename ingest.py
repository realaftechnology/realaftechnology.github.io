"""
ingest.py - Parse Word doc transcripts and store in SQLite with OpenAI embeddings.
Drop .docx files into the /transcripts folder and run this script.
"""

import os, re, sys, json, sqlite3, time, hashlib
from docx import Document

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "db.sqlite"))
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS deleted_episodes (
            id         TEXT PRIMARY KEY,
            deleted_at TEXT
        )
    """)
    # Add file_hash column to existing databases that predate this change
    try:
        cur.execute("ALTER TABLE episodes ADD COLUMN file_hash TEXT")
    except:
        pass
    # published_at: the date the content was originally published (distinct from
    # uploaded_at, which is when the file was added to AFBrain). Used by the
    # Andygram sidebar to show the publish date under the title. ISO YYYY-MM-DD.
    try:
        cur.execute("ALTER TABLE episodes ADD COLUMN published_at TEXT")
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


_MONTHS = {m: i for i, m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

def extract_publish_date(raw_text):
    """Pull the 'Published: Mon DD, YYYY' line out of an Andygram docx.

    Returns an ISO date string (YYYY-MM-DD) or None. The converter writes this
    line as italic metadata right after the title; we scan the first handful
    of non-empty lines to stay cheap.
    """
    for line in raw_text.split("\n")[:6]:
        m = re.match(r'^\s*Published:\s*([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{4})\s*$', line)
        if m:
            mo = _MONTHS.get(m.group(1))
            if not mo:
                return None
            try:
                return f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(2)):02d}"
            except ValueError:
                return None
    return None


def parse_andygram(text):
    """
    Parse an Andygram (Andy's email/blog post) docx into segments.

    Andygrams are prose — no speaker labels, no timestamps. We synthesize a
    single speaker ("Andy") at [00:00] for every body paragraph so downstream
    chunking/embedding works identically to transcripts.

    Skips the first bold line (the title) and any leading "Published:" line.
    """
    segments = []
    saw_title = False
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # First bold line is the title — skip it (caller uses it for episode.title).
        if not saw_title and line.startswith("**") and line.endswith("**"):
            saw_title = True
            continue
        saw_title = True  # After the first non-empty line, never skip-as-title again.
        # Drop the italicized "Published: ..." metadata line.
        if line.lower().startswith("published:"):
            continue
        # Strip stray bold markers inside body text (rare, but possible).
        clean = re.sub(r'\*\*', '', line)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if not clean:
            continue
        segments.append({
            "speaker":    "Andy",
            "timestamp":  "00:00",
            "start_secs": 0.0,
            "text":       clean,
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


def get_embeddings_batch(texts, retries=3):
    """Get OpenAI embeddings for a list of texts in a single API call."""
    if not OPENAI_API_KEY or not texts:
        return [None] * len(texts)

    import urllib.request, json as jsonlib

    # Truncate each text to 8000 chars (well within token limit)
    inputs = [t[:8000] for t in texts]

    payload = jsonlib.dumps({
        "model": "text-embedding-3-small",
        "input": inputs
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = jsonlib.loads(resp.read())
                # API returns results sorted by index
                ordered = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in ordered]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ⚠️  Batch embedding failed: {e}")
                return [None] * len(texts)


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

    # Skip episodes the user has explicitly deleted via the app
    tombstone = cur.execute(
        "SELECT id FROM deleted_episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if tombstone:
        print(f"  🗑️  Deleted by user, skipping: {filename}")
        return 0

    # Check if already ingested with same content
    existing = cur.execute(
        "SELECT id, file_hash, published_at FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if existing:
        if existing[1] == fhash:  # existing[1] = file_hash column
            # Backfill: older rows were ingested before published_at existed.
            # Cheap to peek at the docx now and fill it in without re-embedding.
            if ("ANDYGRAM" in filename.upper() or "ANDYGRAM" in episode_id.upper()) \
                    and not existing[2]:
                pub = extract_publish_date(extract_docx_text(filepath))
                if pub:
                    cur.execute(
                        "UPDATE episodes SET published_at = ? WHERE id = ?",
                        (pub, episode_id),
                    )
                    conn.commit()
                    print(f"  🗓  Backfilled publish date ({pub}): {filename}")
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

    # Parse into segments. Andygrams are prose (no speaker/timestamps) and go
    # through a dedicated parser; everything else is a transcript.
    is_andygram = "ANDYGRAM" in filename.upper() or "ANDYGRAM" in episode_id.upper()
    if is_andygram:
        segments = parse_andygram(raw_text)
    else:
        segments = parse_transcript(raw_text)
    if not segments:
        print(f"  ⚠️  No segments parsed from: {filename}")
        return 0

    # Chunk segments
    chunks = chunk_segments(segments)

    # Andygrams carry a publish date in the body header; store it so the UI
    # can show it beneath the title. None for regular transcripts.
    published_at = extract_publish_date(raw_text) if is_andygram else None

    # Store episode
    cur.execute(
        "INSERT OR REPLACE INTO episodes (id, title, filename, file_hash, published_at) VALUES (?, ?, ?, ?, ?)",
        (episode_id, title, filename, fhash, published_at)
    )

    # Fetch all embeddings in one batch API call
    if OPENAI_API_KEY:
        print(f"    Embedding {len(chunks)} chunks in one batch call...")
        embeddings = get_embeddings_batch([c["text"] for c in chunks])
    else:
        embeddings = [None] * len(chunks)

    # Store chunks
    stored = 0
    for chunk, embedding in zip(chunks, embeddings):
        emb_str = json.dumps(embedding) if embedding else None

        cur.execute("""
            INSERT INTO segments (episode_id, speaker, timestamp, start_secs, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (episode_id, chunk["speaker"], chunk["timestamp"],
              chunk["start_secs"], chunk["text"], emb_str))

        cur.execute(
            "INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)",
            (episode_id, chunk["text"])
        )
        stored += 1

    conn.commit()
    return stored


def main():
    if not os.path.exists(TRANSCRIPTS_DIR):
        os.makedirs(TRANSCRIPTS_DIR)
        print(f"📁 Created '{TRANSCRIPTS_DIR}' folder. Drop your .docx files in there and re-run.")
        sys.exit(0)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Walk recursively so content-type subfolders (e.g. andygrams/) are picked
    # up. Filenames must still be unique since episode_id = basename.
    docx_files = []
    for root, _dirs, files in os.walk(TRANSCRIPTS_DIR):
        for f in files:
            if f.lower().endswith(".docx") and not f.startswith("~$"):
                docx_files.append(os.path.relpath(os.path.join(root, f), TRANSCRIPTS_DIR))

    if not docx_files:
        print(f"No .docx files found in '{TRANSCRIPTS_DIR}' — nothing to ingest.")
        conn.close()
        sys.exit(0)

    if not OPENAI_API_KEY:
        print("⚠️  No OPENAI_API_KEY set — keyword search only, no semantic search.")
        print("    Set it for AI-powered semantic search.\n")

    print(f"Found {len(docx_files)} transcript(s) to ingest\n")

    total = 0
    reingest_happened = False
    for i, relpath in enumerate(sorted(docx_files), 1):
        fpath = os.path.join(TRANSCRIPTS_DIR, relpath)
        print(f"[{i}/{len(docx_files)}] {relpath}")
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
