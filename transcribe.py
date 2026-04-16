"""
transcribe.py — Transcribe an MP3 episode into AFBrain using AssemblyAI
(speaker diarization) + Claude (speaker identification).

Usage:
    python transcribe.py episode.mp3
    python transcribe.py episode.mp3 --episode-id Ep1017 --title "Episode 1017"

Required env vars:
    ASSEMBLYAI_API_KEY   — assemblyai.com (free tier available)
    ANTHROPIC_API_KEY    — for automatic speaker identification
    OPENAI_API_KEY       — optional, enables semantic search embeddings
    DB_PATH              — optional, defaults to db.sqlite
"""

import os, sys, json, time, re, sqlite3, argparse
import urllib.request

ASSEMBLYAI_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY     = os.environ.get("OPENAI_API_KEY", "")
DB_PATH        = os.environ.get("DB_PATH", "db.sqlite")
CHUNK_SIZE     = 400  # words per chunk


# ── DATABASE ──────────────────────────────────────────────────────────────────

def init_db(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id TEXT NOT NULL, speaker TEXT,
        timestamp TEXT, start_secs REAL, text TEXT NOT NULL, embedding TEXT,
        FOREIGN KEY (episode_id) REFERENCES episodes(id))""")
    cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS fts_segments
        USING fts5(episode_id, text, content='')""")
    conn.commit()


# ── ASSEMBLYAI ────────────────────────────────────────────────────────────────

def aai_upload(filepath):
    """Upload audio file to AssemblyAI. Returns upload URL."""
    print(f"  Uploading {os.path.basename(filepath)} ...")
    with open(filepath, "rb") as f:
        data = f.read()
    req = urllib.request.Request(
        "https://api.assemblyai.com/v2/upload",
        data=data,
        headers={
            "authorization": ASSEMBLYAI_KEY,
            "content-type": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["upload_url"]


def aai_submit(upload_url):
    """Submit a transcription job with speaker diarization. Returns job ID."""
    payload = json.dumps({
        "audio_url": upload_url,
        "speaker_labels": True,
    }).encode()
    req = urllib.request.Request(
        "https://api.assemblyai.com/v2/transcript",
        data=payload,
        headers={
            "authorization": ASSEMBLYAI_KEY,
            "content-type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["id"]


def aai_poll(transcript_id):
    """Poll until the transcript job completes. Returns the full result dict."""
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    req = urllib.request.Request(url, headers={"authorization": ASSEMBLYAI_KEY})
    while True:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        status = data["status"]
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
        print(f"    [{status}] waiting 15s ...")
        time.sleep(15)


# ── SPEAKER IDENTIFICATION ────────────────────────────────────────────────────

def identify_speakers(utterances):
    """
    Use Claude Haiku to map speaker labels (A, B, C) to real names.
    Returns dict like {"A": "Andy Frisella", "B": "DJ", "C": "Guest"}.
    Falls back to raw labels if no Anthropic key.
    """
    unique = sorted({u["speaker"] for u in utterances})

    if not ANTHROPIC_KEY:
        print("  No ANTHROPIC_API_KEY — speakers labeled as-is")
        return {s: f"Speaker {s}" for s in unique}

    # Feed the first 40 turns to Claude for identification
    sample = utterances[:40]
    turns  = "\n".join(
        f'Speaker {u["speaker"]}: {u["text"][:250]}' for u in sample
    )

    prompt = f"""These are the first 40 utterances from an Andy Frisella podcast.

The show has two regular hosts:
- Andy Frisella: the main host, entrepreneur, direct and intense, gives long opinionated answers
- DJ: the co-host, shorter responses, often asks questions or sets up topics

There may also be a guest (a third speaker with a different background/voice).

Speaker labels present: {", ".join(unique)}

Transcript sample:
{turns}

Based on speaking style and content, map each label to the correct name.
Return ONLY a JSON object. Use "Andy Frisella", "DJ", or "Guest" (or "Guest: [Name]" if identifiable).
Example: {{"A": "Andy Frisella", "B": "DJ"}}"""

    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 150,
        "temperature": 0,
        "system": "You identify podcast speakers from transcript samples. Return only valid JSON, nothing else.",
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = json.loads(resp.read())["content"][0]["text"].strip()
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        result = json.loads(match.group() if match else text)
        # Ensure every label has a mapping
        for label in unique:
            if label not in result:
                result[label] = f"Speaker {label}"
        return result
    except Exception as e:
        print(f"  Speaker ID failed ({e}) — using raw labels")
        return {s: f"Speaker {s}" for s in unique}


# ── SEGMENT PROCESSING ────────────────────────────────────────────────────────

def ms_to_timestamp(ms):
    s = int(ms) // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def utterances_to_segments(utterances, speaker_map):
    return [
        {
            "speaker":    speaker_map.get(u["speaker"], f"Speaker {u['speaker']}"),
            "timestamp":  ms_to_timestamp(u["start"]),
            "start_secs": u["start"] / 1000.0,
            "text":       u["text"].strip(),
        }
        for u in utterances
        if u.get("text", "").strip()
    ]


def chunk_segments(segments, chunk_size=CHUNK_SIZE):
    chunks = []
    cur_text, cur_ts, cur_secs, cur_spk, word_count = "", "", 0.0, "", 0

    for seg in segments:
        words = seg["text"].split()
        if word_count + len(words) > chunk_size and cur_text:
            chunks.append({"speaker": cur_spk, "timestamp": cur_ts,
                           "start_secs": cur_secs, "text": cur_text.strip()})
            cur_text, word_count = "", 0

        if not cur_text:
            cur_ts, cur_secs, cur_spk = seg["timestamp"], seg["start_secs"], seg["speaker"]

        cur_text  += " " + seg["text"]
        word_count += len(words)

    if cur_text.strip():
        chunks.append({"speaker": cur_spk, "timestamp": cur_ts,
                       "start_secs": cur_secs, "text": cur_text.strip()})
    return chunks


# ── EMBEDDINGS ────────────────────────────────────────────────────────────────

def get_embedding(text, retries=3):
    if not OPENAI_KEY:
        return None
    payload = json.dumps({"model": "text-embedding-3-small", "input": text[:8000]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())["data"][0]["embedding"]
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── DATABASE WRITE ────────────────────────────────────────────────────────────

def ingest_chunks(conn, episode_id, title, filename, chunks):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO episodes (id, title, filename) VALUES (?, ?, ?)",
        (episode_id, title, filename),
    )

    for i, chunk in enumerate(chunks):
        embedding  = get_embedding(chunk["text"])
        emb_json   = json.dumps(embedding) if embedding else None

        cur.execute(
            """INSERT INTO segments (episode_id, speaker, timestamp, start_secs, text, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (episode_id, chunk["speaker"], chunk["timestamp"],
             chunk["start_secs"], chunk["text"], emb_json),
        )
        cur.execute(
            "INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)",
            (episode_id, chunk["text"]),
        )

        if OPENAI_KEY and (i + 1) % 10 == 0:
            print(f"    ... {i + 1}/{len(chunks)} chunks embedded")

    conn.commit()
    return len(chunks)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an MP3 and ingest it into AFBrain"
    )
    parser.add_argument("mp3", help="Path to the MP3 file")
    parser.add_argument("--episode-id", help="Episode ID (default: filename without extension)")
    parser.add_argument("--title",      help="Episode title (default: same as episode ID)")
    args = parser.parse_args()

    if not ASSEMBLYAI_KEY:
        print("Error: ASSEMBLYAI_API_KEY is not set.")
        print("Get a free key at assemblyai.com")
        sys.exit(1)

    if not os.path.exists(args.mp3):
        print(f"Error: File not found: {args.mp3}")
        sys.exit(1)

    filename   = os.path.basename(args.mp3)
    episode_id = args.episode_id or os.path.splitext(filename)[0]
    title      = args.title or episode_id

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    existing = conn.execute(
        "SELECT id FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    if existing:
        print(f"Already ingested: {episode_id}")
        print(f"Use --episode-id to specify a different ID if this is intentional.")
        conn.close()
        sys.exit(0)

    print(f"\nEpisode:  {title}")
    print(f"File:     {filename}")
    print(f"ID:       {episode_id}\n")

    # 1. Upload
    upload_url = aai_upload(args.mp3)
    print(f"  Upload complete.\n")

    # 2. Submit job
    print("  Submitting transcription job (speaker diarization on) ...")
    transcript_id = aai_submit(upload_url)
    print(f"  Job ID: {transcript_id}\n")

    # 3. Wait for result
    print("  Transcribing ... (this takes roughly 15-20% of audio duration)")
    result     = aai_poll(transcript_id)
    utterances = result.get("utterances", [])
    if not utterances:
        print("  Error: No utterances in response. Check AssemblyAI dashboard.")
        sys.exit(1)
    print(f"  Transcription complete — {len(utterances)} utterances\n")

    # 4. Identify speakers
    print("  Identifying speakers ...")
    speaker_map = identify_speakers(utterances)
    for label, name in sorted(speaker_map.items()):
        print(f"    Speaker {label} → {name}")
    print()

    # 5. Convert + chunk
    segments = utterances_to_segments(utterances, speaker_map)
    chunks   = chunk_segments(segments)
    print(f"  {len(segments)} utterances → {len(chunks)} chunks\n")

    # 6. Embed + store
    print("  Ingesting into database ...")
    if not OPENAI_KEY:
        print("  (OPENAI_API_KEY not set — skipping embeddings, keyword search only)\n")
    count = ingest_chunks(conn, episode_id, title, filename, chunks)
    conn.close()

    print(f"\nDone. {count} chunks stored for '{title}'")


if __name__ == "__main__":
    main()
