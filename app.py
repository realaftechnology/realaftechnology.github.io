"""
AFBrain - Flask backend
"""

import os
import json
import sqlite3
import re
import urllib.request
from flask import Flask, request, jsonify, session, send_from_directory
from functools import wraps

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "afbrain-secret-change-this")

DB_PATH    = os.environ.get("DB_PATH", "db.sqlite")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")


# ── AUTH ──────────────────────────────────────────────────────────────────────

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/api/login", methods=["POST"])
def login():
    password = os.environ.get("AFBRAIN_PASSWORD", "CHANGEME")
    data = request.get_json()
    if data.get("password") == password:
        session["authenticated"] = True
        return jsonify({"ok": True})
    return jsonify({"error": "Invalid password"}), 403


@app.route("/api/debug-pw")
def debug_pw():
    return jsonify({"pw": os.environ.get("AFBRAIN_PASSWORD"), "default": "CHANGEME"})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# ── DB HELPERS ────────────────────────────────────────────────────────────────

def get_db():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_episode_count():
    conn = get_db()
    if not conn: return 0
    c = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    conn.close()
    return c


def get_segment_count():
    conn = get_db()
    if not conn: return 0
    c = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    conn.close()
    return c


def has_embeddings():
    conn = get_db()
    if not conn: return False
    c = conn.execute("SELECT COUNT(*) FROM segments WHERE embedding IS NOT NULL").fetchone()[0]
    conn.close()
    return c > 0


# ── SEARCH ────────────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0: return 0.0
    return dot / (mag_a * mag_b)


def get_query_embedding(query):
    if not OPENAI_KEY: return None
    payload = json.dumps({"model": "text-embedding-3-small", "input": query}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())["data"][0]["embedding"]
    except:
        return None


def semantic_search(query, limit=50):
    query_emb = get_query_embedding(query)
    if not query_emb: return []
    conn = get_db()
    if not conn: return []
    rows = conn.execute("""
        SELECT s.id, s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text, s.embedding,
               e.title AS episode_title, e.filename
        FROM segments s JOIN episodes e ON e.id = s.episode_id
        WHERE s.embedding IS NOT NULL
    """).fetchall()
    conn.close()
    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
            score = cosine_similarity(query_emb, emb)
            scored.append((score, row))
        except: continue
    scored.sort(key=lambda x: x[0], reverse=True)
    return [format_result(r, score) for score, r in scored[:limit]]


def keyword_search(query, limit=50):
    conn = get_db()
    if not conn: return []
    try:
        rows = conn.execute("""
            SELECT s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text,
                   e.title AS episode_title, e.filename, fts.rank
            FROM fts_segments fts
            JOIN segments s ON s.rowid = fts.rowid
            JOIN episodes e ON e.id = s.episode_id
            WHERE fts_segments MATCH ?
            ORDER BY fts.rank LIMIT ?
        """, (query, limit)).fetchall()
    except:
        rows = conn.execute("""
            SELECT s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text,
                   e.title AS episode_title, e.filename, 0 AS rank
            FROM segments s JOIN episodes e ON e.id = s.episode_id
            WHERE s.text LIKE ? LIMIT ?
        """, (f"%{query}%", limit)).fetchall()
    conn.close()
    return [format_result(r, 0) for r in rows]


def episode_search(ep_number):
    """Return ALL chunks from a specific episode in chronological order."""
    conn = get_db()
    if not conn: return []
    rows = conn.execute("""
        SELECT s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text,
               e.title AS episode_title, e.filename, 0 AS rank
        FROM segments s JOIN episodes e ON e.id = s.episode_id
        WHERE e.title LIKE ?
        ORDER BY s.start_secs ASC
    """, (f"%{ep_number}%",)).fetchall()
    conn.close()
    return [format_result(r, 0) for r in rows]


def format_result(row, score):
    return {
        "episode_title": row["episode_title"],
        "filename":      row["filename"] or "",
        "speaker":       row["speaker"] or "",
        "timestamp":     row["timestamp"] or "00:00",
        "start_secs":    row["start_secs"] or 0,
        "text":          row["text"],
        "score":         round(float(score), 4),
    }


def sort_by_episode(results):
    def ep_num(r):
        title = r.get("episode_title", "")
        m = re.search(r'(?:Ep|EP|ep)[\s_-]*(\d+)', title)
        if m: return int(m.group(1))
        m = re.search(r'(\d{3,4})', title)
        if m: return int(m.group(1))
        return 0
    return sorted(results, key=ep_num, reverse=True)


def ai_rerank(query, candidates, top_n=20):
    if not OPENAI_KEY or not candidates: return candidates[:top_n]
    context = ""
    for i, c in enumerate(candidates[:30]):
        context += f"[{i}] {c['episode_title']} | {c['timestamp']}\n{c['text'][:200]}\n\n"
    prompt = f'Search: "{query}"\n\nCandidates:\n{context}\nReturn a JSON array of the {top_n} most relevant indices, ordered by relevance. Only the array, nothing else.'
    payload = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
            indices = json.loads(text)
            return [candidates[i] for i in indices if i < len(candidates)]
    except:
        return candidates[:top_n]


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@app.route("/api/stats")
@requires_auth
def stats():
    return jsonify({
        "episodes": get_episode_count(),
        "segments": get_segment_count(),
        "has_embeddings": has_embeddings(),
        "has_openai": bool(OPENAI_KEY),
        "mode": "semantic" if (has_embeddings() and OPENAI_KEY) else "keyword"
    })


@app.route("/api/search", methods=["POST"])
@requires_auth
def search_endpoint():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"results": []})

    # Detect episode number reference — if found, return only that episode
    ep_match = re.search(r'\b(ep|episode)[\s_-]*(\d{3,4})\b', query, re.IGNORECASE)
    if not ep_match:
        ep_match = re.search(r'\b(\d{4})\b', query)
    ep_filter = ep_match.group(2) if ep_match and len(ep_match.groups()) > 1 else (ep_match.group(1) if ep_match else None)

    if ep_filter:
        results = episode_search(ep_filter)
    elif has_embeddings() and OPENAI_KEY:
        results = semantic_search(query, limit=100)
        if not results:
            results = keyword_search(query, limit=100)
        if OPENAI_KEY:
            results = ai_rerank(query, results, top_n=50)
    else:
        results = keyword_search(query, limit=100)

    results = sort_by_episode(results)
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/analyze", methods=["POST"])
@requires_auth
def analyze():
    if not OPENAI_KEY:
        return jsonify({"error": "No OpenAI key configured"})
    data = request.get_json()
    query   = data.get("query", "")
    results = data.get("results", [])  # No limit — use all results passed

    context = "\n\n---\n\n".join([f"[{r['episode_title']} @ {r['timestamp']}]\n{r['text']}" for r in results])
    prompt = f'''You are an AI assistant with access to Andy Frisella podcast transcripts.

The user asked: "{query}"

Here are the relevant transcript excerpts:
{context}

Answer the user's question directly and specifically using only these transcripts.

- If they ask what happened in a specific episode or segment, summarize it clearly
- If they ask Andy to tell a story, piece together the narrative from all mentions across episodes
- If they ask for specific content (headline 3, prediction, story), find and summarize it
- Always cite the episode and timestamp for each key point
- If the transcripts don't contain enough information to answer, say so clearly

Be direct and specific. Answer the question they actually asked.'''

    payload = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500, "temperature": 0.2}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = json.loads(resp.read())["choices"][0]["message"]["content"]
            return jsonify({"analysis": text})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/followup", methods=["POST"])
@requires_auth
def followup():
    if not OPENAI_KEY:
        return jsonify({"error": "No OpenAI key configured"})
    data         = request.get_json()
    question     = data.get("question", "")
    chat_history = data.get("history", [])

    # Check for episode reference in follow-up
    ep_match = re.search(r'\b(ep|episode)[\s_-]*(\d{3,4})\b', question, re.IGNORECASE)
    ep_filter = ep_match.group(2) if ep_match else None

    if ep_filter:
        fu_results = episode_search(ep_filter)
    elif has_embeddings() and OPENAI_KEY:
        fu_results = semantic_search(question, limit=20)
    else:
        fu_results = keyword_search(question, limit=20)

    context = "\n\n---\n\n".join([f"[{r['episode_title']} @ {r['timestamp']}]\n{r['text']}" for r in fu_results]) if fu_results else "No relevant transcript excerpts found."

    messages = [{"role": "system", "content": "You answer questions about Andy Frisella's podcasts using only provided transcripts. Answer the specific question asked. Cite quotes and episodes. If nothing relevant, say so directly."}]
    for item in chat_history:
        messages.append({"role": "user", "content": item["q"]})
        messages.append({"role": "assistant", "content": item["a"]})
    messages.append({"role": "user", "content": f'Question: "{question}"\n\nTranscripts:\n{context}\n\nAnswer the specific question asked. Cite quotes with episode and timestamp.'})

    payload = json.dumps({"model": "gpt-4o-mini", "messages": messages, "max_tokens": 600, "temperature": 0.2}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = json.loads(resp.read())["choices"][0]["message"]["content"]
            return jsonify({"answer": text})
    except Exception as e:
        return jsonify({"error": str(e)})


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)