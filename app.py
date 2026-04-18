"""
AFBrain - Flask backend
"""

import os
import json
import sqlite3
import re
import subprocess
import shutil
import tempfile
import time
import threading
import queue as queue_module
import uuid
import urllib.request
import urllib.error
from collections import deque
from datetime import datetime
from flask import Flask, request, jsonify, session, send_from_directory, send_file, Response, stream_with_context, after_this_request, render_template_string
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "afbrain-secret-change-this")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10GB max upload

DB_PATH       = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "db.sqlite"))
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")   # used for Whisper transcription + embeddings
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "") # used for all AI text generation

# ── Import job registry (in-memory, last 100 jobs) ────────────────────────────
_jobs      = deque(maxlen=100)
_jobs_lock = threading.Lock()

def _new_job(urls):
    job = {
        "id":          uuid.uuid4().hex[:8],
        "started_at":  datetime.now().isoformat(),
        "finished_at": None,
        "urls":        urls,
        "status":      "running",
        "logs":        [],
        "files":       [],   # per-file progress: [{name, status, started_at, finished_at, segments}]
        "stats":       {"done": 0, "skipped": 0, "errors": 0},
    }
    with _jobs_lock:
        _jobs.appendleft(job)
    return job

def _job_log(job, msg_type, msg, msg_q=None):
    entry = {"ts": datetime.now().strftime("%H:%M:%S"), "type": msg_type, "msg": msg}
    with _jobs_lock:
        job["logs"].append(entry)
        if msg_type == "done_file": job["stats"]["done"]    += 1
        elif msg_type == "skipped": job["stats"]["skipped"] += 1
        elif msg_type == "error":   job["stats"]["errors"]  += 1
    if msg_q is not None:
        msg_q.put({"type": msg_type, "msg": msg})

def _finish_job(job, status="done"):
    with _jobs_lock:
        job["status"]      = status
        job["finished_at"] = datetime.now().isoformat()

YOUTUBE_TITLE_KNOWLEDGE = """
YOUTUBE TITLE GENERATION EXPERTISE:

When asked to generate YouTube titles, apply the following framework:

STEP 1 — IDENTIFY HIGH-VALUE MOMENTS
Scan the transcript for these high-click-potential content types:
- Counterintuitive truths: Andy says something that flips a common belief
- Hard personal stories: failure, loss, conflict, turning points
- Polarizing opinions: things he says that will make some people uncomfortable
- Specific tactical advice that challenges what most people do
- Named frameworks, rules, or concepts Andy introduces
- Emotional peaks: raw anger, vulnerability, certainty, urgency
- Any moment where Andy says what other people in the space won't say

STEP 2 — TITLE PRINCIPLES
Structure: Front-load the hook. The first 3–4 words decide if anyone reads the rest.
Length: 50–65 characters ideal. Cut every word that doesn't earn its place.
Voice: Andy's tone is direct, blunt, anti-excuse, pro-accountability. Zero corporate speak.
Audience: Entrepreneurs, builders, people who want brutal honesty — not cheerleading.

FORMATTING RULES — THESE ARE HARD RULES:
- NO exclamation points. They signal desperation and reduce perceived credibility.
- NO em-dashes (—) in titles. They break rhythm and look cluttered. Use a period or rewrite as one clean thought.
- NO all-caps words. Use sentence case or title case only.
- NO generic hype words: game-changer, amazing, incredible, life-changing, powerful.
- NO vague words: things, stuff, this, it.
- Use present tense — creates immediacy.
- Use "you" or implied second person — makes it personal.
- Specificity always beats vagueness. Name the exact topic, belief, or mistake.

STEP 3 — TITLE FORMULAS (pick the best fit for the content)
Tension/stakes:      "The Mistake That Almost [Serious Consequence]"
Blunt reframe:       "[Positive-Sounding Thing] Is Actually Holding You Back"
Direct challenge:    "You're Not [Positive Identity]. You're [Uncomfortable Truth]."
Counterintuitive:    "Why [Common Advice] Keeps Most People [Stuck/Broke/Weak]"
Uncomfortable truth: "Nobody Talks About [Uncomfortable Side of Topic]"
Specific experience: "What [X Years / Specific Event] Taught Me About [Topic]"
Named concept:       "The [Andy's Term]: Why [Most People] Never [Goal]"
Hard question:       "Are You Building [Thing] or Just Telling Yourself You Are"
Stated truth:        "[Contrarian Claim]. Here's the Proof."
Pattern break:       "I Was [Common Belief] For [X] Years. I Was Wrong."

STEP 4 — OUTPUT FORMAT
Generate 8–10 title options. Group by style (e.g., Tension, Reframe, Direct Challenge).
After the list, call out the 2–3 BEST picks with one sentence on why each will perform.
Every title must be grounded in what is actually in the transcript — no fabricated topics.
"""


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
    # Migrations for older dbs
    for stmt in [
        "ALTER TABLE episodes ADD COLUMN video_path TEXT",
        "ALTER TABLE episodes ADD COLUMN uploaded_at TEXT",
        "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)",
        """CREATE TABLE IF NOT EXISTS search_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            query     TEXT NOT NULL,
            ep_filter TEXT,
            results   INTEGER DEFAULT 0,
            searched_at TEXT DEFAULT (datetime('now'))
        )""",
        # Tombstone: prevents ingest.py from re-adding episodes deleted via the app
        """CREATE TABLE IF NOT EXISTS deleted_episodes (
            id         TEXT PRIMARY KEY,
            deleted_at TEXT
        )""",
    ]:
        try:
            conn.execute(stmt)
            conn.commit()
        except Exception:
            pass
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


def semantic_search(query, limit=50, episode_filter=None):
    query_emb = get_query_embedding(query)
    if not query_emb: return []
    conn = get_db()
    if not conn: return []
    if episode_filter:
        where, params = ep_filter_clauses(episode_filter)
        rows = conn.execute(f"""
            SELECT s.id, s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text, s.embedding,
                   e.title AS episode_title, e.filename, e.video_path
            FROM segments s JOIN episodes e ON e.id = s.episode_id
            WHERE s.embedding IS NOT NULL AND {where}
        """, params).fetchall()
    else:
        rows = conn.execute("""
            SELECT s.id, s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text, s.embedding,
                   e.title AS episode_title, e.filename, e.video_path
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
                   e.title AS episode_title, e.filename, e.video_path, fts.rank
            FROM fts_segments fts
            JOIN segments s ON s.rowid = fts.rowid
            JOIN episodes e ON e.id = s.episode_id
            WHERE fts_segments MATCH ?
            ORDER BY fts.rank LIMIT ?
        """, (query, limit)).fetchall()
    except:
        rows = conn.execute("""
            SELECT s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text,
                   e.title AS episode_title, e.filename, e.video_path, 0 AS rank
            FROM segments s JOIN episodes e ON e.id = s.episode_id
            WHERE s.text LIKE ? LIMIT ?
        """, (f"%{query}%", limit)).fetchall()
    conn.close()
    return [format_result(r, 0) for r in rows]


def ep_filter_clauses(ep_number):
    """Return (sql_fragment, params) that precisely matches an episode number.
    Avoids 'Ep 1' matching 'Ep 10', 'Ep 100', 'Ep 1000' etc."""
    try:
        n = int(ep_number)
        patterns = [f"Ep {n} %", f"Ep {n}-%", f"Ep {n}.%", f"Ep {n}"]
        if n < 100:
            p = f"{n:02d}"
            patterns += [f"Ep {p} %", f"Ep {p}-%", f"Ep {p}.%"]
        sql = "(" + " OR ".join(["e.title LIKE ?"] * len(patterns)) + ")"
        return sql, patterns
    except ValueError:
        # Non-numeric filter: match against title OR episode id (which may have
        # underscores where the title has spaces — Drive imports hit this case)
        normalized = ep_number.replace("_", " ").replace("-", " ")
        sql = "(e.title LIKE ? OR e.id LIKE ? OR e.title LIKE ?)"
        return sql, [f"%{ep_number}%", f"%{ep_number}%", f"%{normalized}%"]


def episode_search(ep_number):
    """Return ALL chunks from a specific episode in chronological order."""
    conn = get_db()
    if not conn: return []
    where, params = ep_filter_clauses(ep_number)
    rows = conn.execute(f"""
        SELECT s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text,
               e.title AS episode_title, e.filename, e.video_path, 0 AS rank
        FROM segments s JOIN episodes e ON e.id = s.episode_id
        WHERE {where}
        ORDER BY s.start_secs ASC
    """, params).fetchall()
    conn.close()
    return [format_result(r, 0) for r in rows]


def clean_ep_label(episode_title):
    """Extract a short readable label from a filename-based episode title."""
    m = re.search(r'(\d+)', episode_title)
    return f"Ep {m.group(1)}" if m else episode_title


def format_result(row, score):
    video_path = row["video_path"] if "video_path" in row.keys() else None
    return {
        "episode_title":  row["episode_title"],
        "filename":       row["filename"] or "",
        "speaker":        row["speaker"] or "",
        "timestamp":      row["timestamp"] or "00:00",
        "start_secs":     row["start_secs"] or 0,
        "text":           row["text"],
        "score":          round(float(score), 4),
        "has_video":      bool(video_path),
        "video_filename": os.path.basename(video_path) if video_path else None,
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


def anthropic_call(system, messages, model, max_tokens, temperature=0.3):
    """Call the Anthropic Messages API. Returns response text or raises on error."""
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())["content"][0]["text"]


def anthropic_stream(system, messages, model, max_tokens, temperature=0.3):
    """Generator that yields SSE chunks from Anthropic streaming API."""
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("event:") and "ping" in line:
                    yield ": ping\n\n"
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                    if event.get("type") == "content_block_delta":
                        text = event["delta"].get("text", "")
                        if text:
                            yield f"data: {json.dumps({'text': text})}\n\n"
                except Exception:
                    pass
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        yield f"data: {json.dumps({'error': f'API error {e.code}: {body}'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


def ai_rerank(query, candidates, top_n=20):
    if not ANTHROPIC_KEY or not candidates: return candidates[:top_n]
    context = ""
    for i, c in enumerate(candidates[:30]):
        context += f"[{i}] {c['episode_title']} | {c['timestamp']}\n{c['text'][:200]}\n\n"
    user_msg = f'Search: "{query}"\n\nCandidates:\n{context}\nReturn a JSON array of the {top_n} most relevant indices, ordered by relevance. Only the array, nothing else.'
    try:
        text = anthropic_call(
            system="You are a search relevance ranker. Return only a valid JSON array of integers. No explanation.",
            messages=[{"role": "user", "content": user_msg}],
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            temperature=0,
        )
        indices = json.loads(text.strip())
        return [candidates[i] for i in indices if i < len(candidates)]
    except:
        return candidates[:top_n]


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@app.route("/api/episodes")
@requires_auth
def episodes_list():
    conn = get_db()
    if not conn: return jsonify({"episodes": []})
    rows = conn.execute("""
        SELECT e.id, e.title, e.video_path, e.uploaded_at, COUNT(s.id) as segment_count
        FROM episodes e LEFT JOIN segments s ON s.episode_id = e.id
        GROUP BY e.id ORDER BY e.id DESC
    """).fetchall()
    conn.close()
    return jsonify({"episodes": [
        {"id": r["id"], "title": r["title"], "segments": r["segment_count"],
         "has_video": bool(r["video_path"]),
         "video_filename": os.path.basename(r["video_path"]) if r["video_path"] else None,
         "uploaded_at": r["uploaded_at"]}
        for r in rows
    ]})


@app.route("/api/stats")
@requires_auth
def stats():
    return jsonify({
        "episodes": get_episode_count(),
        "segments": get_segment_count(),
        "has_embeddings": has_embeddings(),
        "has_openai": bool(OPENAI_KEY),
        "has_claude": bool(ANTHROPIC_KEY),
        "mode": "semantic" if (has_embeddings() and OPENAI_KEY) else "keyword"
    })


@app.route("/api/search", methods=["POST"])
@requires_auth
def search_endpoint():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"results": []})

    # Explicit episode filter from UI (sidebar) takes priority over query detection
    ep_filter = data.get("episode_filter", "") or extract_episode_filter(query)

    if ep_filter:
        results = episode_search(ep_filter)
    elif has_embeddings() and OPENAI_KEY:
        results = semantic_search(query, limit=100)
        if not results:
            results = keyword_search(query, limit=100)
        if ANTHROPIC_KEY:
            results = ai_rerank(query, results, top_n=50)
    else:
        results = keyword_search(query, limit=100)

    results = sort_by_episode(results)

    # Log to search history (best-effort, never block the response)
    try:
        conn = get_db()
        if conn:
            conn.execute(
                "INSERT INTO search_history (query, ep_filter, results) VALUES (?, ?, ?)",
                (query, ep_filter or None, len(results))
            )
            conn.commit()
            conn.close()
    except Exception:
        pass

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/analyze", methods=["POST"])
@requires_auth
def analyze():
    if not ANTHROPIC_KEY:
        return jsonify({"error": "No Anthropic API key configured"})
    data    = request.get_json()
    query   = data.get("query", "")
    results = data.get("results", [])

    context = "\n\n---\n\n".join([f"[{clean_ep_label(r['episode_title'])} @ {r['timestamp']}]\n{r['text']}" for r in results])

    system = f"""You are an AI assistant for Andy Frisella's internal team, built on his podcast transcripts.

CORE BEHAVIOR:
You always produce a real, useful response. Never return empty. Never say "I can't help with that."

HOW TO HANDLE QUERIES:
1. Interpret what the user is most likely asking. If a question is ambiguous, address the most likely interpretations — e.g. "If you're asking about X: [answer]. If you're asking about Y: [answer]."
2. Every specific claim — names, relationships, events, statements — must come verbatim from the transcript excerpts provided. Do not draw on outside knowledge to fill in details.
3. If the transcripts cover the topic well, answer fully and directly.
4. If coverage is thin, say exactly what you found and suggest specific search terms to go deeper. Do not pad the answer with plausible-sounding details.
5. Themes and inferences are fine ("Andy frequently ties discipline to identity"). Specific facts are not, unless they appear in the excerpts ("Andy's son Enzo" is wrong unless those words are in the text).

CONTENT CREATION:
If asked to draft tweets, social copy, show notes, scripts, YouTube titles, or anything else — do it, grounded in Andy's actual words and arguments from the transcripts.

WHAT TO SKIP:
- Guest introductions, show format names (CTI/Q&AF), recurring segment names — team already knows
- Generic observations that apply to every episode — only surface what's specific and unique

CITATIONS:
- Prose answers: no blockquote needed
- When exact wording matters: > "Quote" — Ep 1014, 00:10:55
- Cite sparingly. No horizontal rules. Bold labels, not headers.
{YOUTUBE_TITLE_KNOWLEDGE}"""

    user_msg = f'Query: "{query}"\n\nTranscript excerpts:\n{context}'

    return Response(
        stream_with_context(anthropic_stream(
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0.4,
        )),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def extract_episode_filter(text):
    """Extract an episode number from text. Returns string or None."""
    m = re.search(r'\b(?:ep|episode)[\s_-]*(\d+)\b', text, re.IGNORECASE)
    if m: return m.group(1)
    # Match 4-digit numbers but exclude calendar years (2000-2030) which appear
    # in date-based Drive titles like "04.12.2024 Audio Exclusive"
    for m in re.finditer(r'\b(\d{4})\b', text):
        if not (2000 <= int(m.group(1)) <= 2030):
            return m.group(1)
    return None


@app.route("/api/followup", methods=["POST"])
@requires_auth
def followup():
    if not ANTHROPIC_KEY:
        return jsonify({"error": "No Anthropic API key configured"})
    data           = request.get_json()
    question       = data.get("question", "")
    chat_history   = data.get("history", [])
    original_query = data.get("original_query", "")

    ep_in_question = extract_episode_filter(question)
    ep_in_original = extract_episode_filter(original_query)

    # If the last AI answer mentioned a specific episode, treat it as context for follow-ups
    ep_in_last_answer = None
    if chat_history:
        ep_in_last_answer = extract_episode_filter(chat_history[-1].get("a", ""))

    q_lower = question.lower()

    # Detect "next episode" / "episode after" — advance from last known episode
    next_ep_request = any(p in q_lower for p in ['next episode', 'episode after', 'following episode'])
    prev_ep_request = any(p in q_lower for p in ['previous episode', 'episode before', 'last episode'])

    # Detect when user wants to expand search beyond the current episode
    cross_episode_request = any(w in q_lower for w in [
        'subsequent', 'other episode', 'later episode', 'different episode',
        'across episode', 'another episode', 'other show', 'expand', 'broader',
        'elsewhere', 'any other', 'more episode'
    ])

    if next_ep_request and ep_in_last_answer:
        try:
            ep_filter = str(int(ep_in_last_answer) + 1)
        except:
            ep_filter = ep_in_last_answer
    elif prev_ep_request and ep_in_last_answer:
        try:
            ep_filter = str(max(1, int(ep_in_last_answer) - 1))
        except:
            ep_filter = ep_in_last_answer
    elif cross_episode_request:
        ep_filter = data.get("episode_filter", "") or ep_in_question
    else:
        ep_filter = data.get("episode_filter", "") or ep_in_question or ep_in_original or ep_in_last_answer

    # Vague follow-up: no new episode introduced, and asking for another/different
    vague_followup = (
        not cross_episode_request and
        (not ep_in_question or ep_in_question == ep_in_original) and
        any(w in question.lower() for w in ['another', 'different', 'more', 'else', 'new one'])
    )

    if next_ep_request or prev_ep_request:
        # Navigating episodes — use original topic as search query in new episode
        search_query = original_query or question
    elif vague_followup:
        # "Give me another" — search original topic in same episode
        search_query = original_query or question
    elif ep_in_question and ep_in_question != ep_in_original and original_query:
        # New episode introduced ("what about 1016?") — use original topic in new episode
        search_query = original_query
    elif original_query:
        # Clarifying/pushback follow-up — combine original topic + new specifics for best coverage
        search_query = f"{original_query} {question}"
    else:
        search_query = question

    if ep_filter and has_embeddings() and OPENAI_KEY:
        fu_results = semantic_search(search_query, limit=25, episode_filter=ep_filter)
        if not fu_results:
            fu_results = episode_search(ep_filter)[:25]
    elif ep_filter:
        fu_results = episode_search(ep_filter)[:25]
    elif has_embeddings() and OPENAI_KEY:
        fu_results = semantic_search(search_query, limit=20)
    else:
        fu_results = keyword_search(search_query, limit=20)

    # Note: semantic search still uses OpenAI embeddings; only text generation uses Claude

    context = "\n\n---\n\n".join([f"[{clean_ep_label(r['episode_title'])} @ {r['timestamp']}]\n{r['text']}" for r in fu_results]) if fu_results else "No relevant transcript excerpts found."

    # Only inject blocklist for vague "give me another" requests — not when switching topics/episodes
    already_shown = ""
    if chat_history and vague_followup:
        prev = "\n\n".join([f"- {item['a']}" for item in chat_history])
        already_shown = f"\n\nQUOTES ALREADY GIVEN — DO NOT REPEAT ANY OF THESE:\n{prev}"

    system_prompt = (
        "You are an AI assistant for Andy Frisella's internal team, built on his podcast transcripts.\n\n"
        "CORE BEHAVIOR: Always produce a real, useful response. Never return empty.\n\n"
        "HOW TO HANDLE QUERIES:\n"
        "1. Interpret what the user is most likely asking. If ambiguous, address the likely interpretations.\n"
        "2. Every specific claim — names, relationships, events, statements — must come verbatim from the transcript excerpts. Do not draw on outside knowledge to fill in details.\n"
        "3. Themes and inferences are fine. Specific facts are not, unless they appear in the excerpts.\n"
        "4. If coverage is thin, say exactly what you found and give specific search suggestions. Do not pad with plausible-sounding details.\n"
        "5. If asked for content (tweets, copy, scripts, titles) — create it from Andy's actual words.\n\n"
        "CRITICAL: Each excerpt is labeled [Episode Title @ Timestamp]. "
        "Never attribute a quote to a different episode than its label.\n\n"
        "CITATIONS: Prose answers need no blockquote. When exact wording matters: "
        "> \"Quote\" — Ep 1014, 00:10:55. Cite sparingly. No horizontal rules.\n\n"
        + YOUTUBE_TITLE_KNOWLEDGE
    )

    messages = []
    for item in chat_history:
        messages.append({"role": "user", "content": item["q"]})
        messages.append({"role": "assistant", "content": item["a"]})
    messages.append({"role": "user", "content": f'Question: "{question}"{already_shown}\n\nTranscripts:\n{context}\n\nDo NOT repeat any quote listed above.'})

    return Response(
        stream_with_context(anthropic_stream(
            system=system_prompt,
            messages=messages,
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0.4,
        )),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── VIDEO ─────────────────────────────────────────────────────────────────────

def videos_dir():
    return os.path.join(os.path.dirname(DB_PATH), "videos")


def temp_uploads_dir():
    return os.path.join(os.path.dirname(DB_PATH), "temp_uploads")


def secs_to_timestamp(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def extract_audio(video_path, audio_path):
    """Extract audio from video to mp3 using ffmpeg. Returns False if ffmpeg unavailable."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ab", "64k", "-y", audio_path],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
        return True
    except FileNotFoundError:
        return False  # ffmpeg not installed — caller will send video directly


def whisper_transcribe(audio_path):
    """Transcribe a single audio file via OpenAI Whisper API. Must be ≤25MB."""
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set — cannot transcribe")

    with open(audio_path, "rb") as f:
        audio_data = f.read()
    boundary = "Boundary" + os.urandom(8).hex()
    filename = os.path.basename(audio_path)

    def part(name, value):
        return (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()

    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: audio/mpeg\r\n\r\n"
    ).encode() + audio_data + b"\r\n"
    body += part("model", "whisper-1")
    body += part("response_format", "verbose_json")
    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=body,
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    max_retries = 4
    base_delay  = 15  # seconds; doubles each attempt: 15, 30, 60, 120
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[whisper] rate-limited (429) — waiting {delay}s (retry {attempt + 2}/{max_retries})",
                      flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Whisper API error {e.code}: {e.reason}")
    raise RuntimeError("Whisper API: max retries exceeded after rate limiting")


def whisper_transcribe_chunked(audio_path):
    """Transcribe audio, automatically splitting into chunks if it exceeds Whisper's 25MB limit."""
    file_size = os.path.getsize(audio_path)
    if file_size <= 24 * 1024 * 1024:
        return whisper_transcribe(audio_path)

    # Get total duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path],
        capture_output=True, timeout=30
    )
    info = json.loads(probe.stdout)
    total_duration = float(info["format"]["duration"])

    # Split into ~22MB chunks (≈45 min at 64kbps)
    chunk_secs = 2700.0  # 45 minutes
    num_chunks = int(total_duration / chunk_secs) + 1

    all_segments = []
    chunk_paths = []
    try:
        for i in range(num_chunks):
            start = i * chunk_secs
            if start >= total_duration:
                break
            chunk_path = audio_path.replace(".mp3", f"_c{i}.mp3")
            chunk_paths.append(chunk_path)
            subprocess.run(
                ["ffmpeg", "-i", audio_path, "-ss", str(start), "-t", str(chunk_secs),
                 "-c", "copy", "-y", chunk_path],
                capture_output=True, timeout=120
            )
            result = whisper_transcribe(chunk_path)
            for seg in result.get("segments", []):
                adjusted = dict(seg)
                adjusted["start"] = float(seg["start"]) + start
                all_segments.append(adjusted)
    finally:
        for cp in chunk_paths:
            try:
                os.remove(cp)
            except Exception:
                pass

    return {"segments": all_segments}


def get_embeddings_batch_local(texts):
    if not OPENAI_KEY or not texts:
        return [None] * len(texts)
    payload = json.dumps({"model": "text-embedding-3-small", "input": [t[:8000] for t in texts]}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return [None] * len(texts)


def ingest_video_episode(conn, episode_id, title, video_path, whisper_result):
    """Store a video episode from Whisper output into the db."""
    cur = conn.cursor()
    cur.execute("DELETE FROM segments WHERE episode_id = ?", (episode_id,))
    cur.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
    cur.execute(
        "INSERT INTO episodes (id, title, filename, video_path, uploaded_at) VALUES (?, ?, ?, ?, ?)",
        (episode_id, title, os.path.basename(video_path), video_path, datetime.now().strftime('%Y-%m-%d'))
    )

    raw = [
        {"timestamp": secs_to_timestamp(seg["start"]), "start_secs": float(seg["start"]), "text": seg["text"].strip()}
        for seg in whisper_result.get("segments", []) if seg.get("text", "").strip()
    ]
    if not raw:
        conn.commit()
        return 0

    # Chunk into ~400-word segments
    chunks, current_text, current_ts, current_secs, word_count = [], "", "", 0.0, 0
    for seg in raw:
        words = seg["text"].split()
        if word_count + len(words) > 400 and current_text:
            chunks.append({"timestamp": current_ts, "start_secs": current_secs, "text": current_text.strip()})
            current_text, word_count = "", 0
        if not current_text:
            current_ts, current_secs = seg["timestamp"], seg["start_secs"]
        current_text += " " + seg["text"]
        word_count += len(words)
    if current_text.strip():
        chunks.append({"timestamp": current_ts, "start_secs": current_secs, "text": current_text.strip()})

    embeddings = get_embeddings_batch_local([c["text"] for c in chunks])
    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO segments (episode_id, speaker, timestamp, start_secs, text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (episode_id, "", chunk["timestamp"], chunk["start_secs"], chunk["text"], json.dumps(emb) if emb else None)
        )
        cur.execute("INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)", (episode_id, chunk["text"]))
    conn.commit()
    return len(chunks)


@app.route("/api/upload-chunk", methods=["POST"])
@requires_auth
def upload_chunk():
    """Receive one chunk of a multipart upload. Stored in temp_uploads/<upload_id>/chunk_NNNNN."""
    upload_id = secure_filename(request.form.get("upload_id", ""))
    chunk_index = request.form.get("chunk_index", "0")
    chunk_file = request.files.get("chunk")
    if not upload_id or not chunk_file:
        return jsonify({"error": "Missing data"}), 400
    upload_dir = os.path.join(temp_uploads_dir(), upload_id)
    os.makedirs(upload_dir, exist_ok=True)
    chunk_file.save(os.path.join(upload_dir, f"chunk_{int(chunk_index):05d}"))
    return jsonify({"ok": True})


@app.route("/api/finalize-upload", methods=["POST"])
@requires_auth
def finalize_upload():
    """Assemble chunks, extract audio, transcribe, and ingest the episode."""
    if not OPENAI_KEY:
        return jsonify({"error": "OpenAI API key required for transcription"}), 400
    data = request.get_json()
    upload_id = secure_filename(data.get("upload_id", ""))
    filename = secure_filename(data.get("filename", "video.mp4"))
    title = data.get("title", "").strip()
    total_chunks = int(data.get("total_chunks", 0))

    upload_dir = os.path.join(temp_uploads_dir(), upload_id)
    if not os.path.isdir(upload_dir):
        return jsonify({"error": "Upload session not found"}), 404

    vdir = videos_dir()
    os.makedirs(vdir, exist_ok=True)
    video_path = os.path.join(vdir, filename)

    # Assemble chunks in order
    try:
        with open(video_path, "wb") as out:
            for i in range(total_chunks):
                chunk_path = os.path.join(upload_dir, f"chunk_{i:05d}")
                with open(chunk_path, "rb") as cp:
                    out.write(cp.read())
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)

    audio_path = video_path.rsplit(".", 1)[0] + "_audio.mp3"
    try:
        ffmpeg_ok = extract_audio(video_path, audio_path)
        if not ffmpeg_ok:
            if os.path.getsize(video_path) > 24 * 1024 * 1024:
                raise RuntimeError("ffmpeg is required to process files larger than 24MB.")
            result = whisper_transcribe(video_path)
        else:
            result = whisper_transcribe_chunked(audio_path)

        episode_id = filename.rsplit(".", 1)[0]
        title = title or episode_id.replace("_", " ").replace("-", " ")
        conn = get_db()
        count = ingest_video_episode(conn, episode_id, title, video_path, result)
        _rebuild_fts(conn)
        conn.commit()
        conn.close()
        return jsonify({"ok": True, "episode_id": episode_id, "chunks": count, "title": title})
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route("/api/upload-video", methods=["POST"])
@requires_auth
def upload_video():
    if not OPENAI_KEY:
        return jsonify({"error": "OpenAI API key required for transcription"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    vdir = videos_dir()
    os.makedirs(vdir, exist_ok=True)
    filename = secure_filename(file.filename)
    video_path = os.path.join(vdir, filename)
    file.save(video_path)

    audio_path = video_path.rsplit(".", 1)[0] + "_audio.mp3"
    try:
        ffmpeg_ok = extract_audio(video_path, audio_path)
        if not ffmpeg_ok:
            # ffmpeg unavailable — only allow small files
            if os.path.getsize(video_path) > 24 * 1024 * 1024:
                raise RuntimeError("ffmpeg is required to process files larger than 24MB.")
            result = whisper_transcribe(video_path)
        else:
            # Chunked transcription handles any length
            result = whisper_transcribe_chunked(audio_path)
        episode_id = filename.rsplit(".", 1)[0]
        title = request.form.get("title", "").strip() or episode_id
        conn = get_db()
        count = ingest_video_episode(conn, episode_id, title, video_path, result)
        _rebuild_fts(conn)
        conn.commit()
        conn.close()
        return jsonify({"ok": True, "episode_id": episode_id, "chunks": count, "title": title})
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": str(e)}), 500
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


@app.route("/api/drive-sources", methods=["GET"])
@requires_auth
def get_drive_sources():
    conn = get_db()
    if not conn:
        return jsonify({"sources": []})
    row = conn.execute("SELECT value FROM settings WHERE key='drive_sources'").fetchone()
    conn.close()
    sources = json.loads(row[0]) if row else []
    return jsonify({"sources": sources})


@app.route("/api/drive-sources", methods=["POST"])
@requires_auth
def save_drive_sources():
    data = request.get_json()
    sources = [s.strip() for s in data.get("sources", []) if s.strip()]
    conn = get_db()
    if not conn:
        return jsonify({"error": "DB not available"}), 500
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES ('drive_sources', ?)",
        (json.dumps(sources),)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "count": len(sources)})


_VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.mp3', '.m4a', '.webm')
_MIN_FREE_BYTES = 6 * 1024 ** 3  # 6 GB safety buffer


def _already_ingested(episode_id):
    """Return True if this episode_id already exists in the DB."""
    try:
        conn = get_db()
        row = conn.execute("SELECT id FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        conn.close()
        return row is not None
    except Exception:
        return False


def _disk_free_bytes():
    """Return free bytes on the data volume (or app dir as fallback)."""
    for path in ("/data", os.path.dirname(DB_PATH), "/"):
        try:
            return shutil.disk_usage(path).free
        except Exception:
            continue
    return float("inf")


def _fmt_bytes(n):
    if n >= 1e9: return f"{n/1e9:.1f} GB"
    if n >= 1e6: return f"{n/1e6:.1f} MB"
    return f"{n/1e3:.0f} KB"


def _list_gdrive_folder(url):
    """
    List video files in a public Google Drive folder WITHOUT downloading them.
    Returns list of {'id': str, 'name': str} dicts.
    Raises RuntimeError on failure.
    """
    m = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if not m:
        raise RuntimeError(f"Cannot extract folder ID from URL: {url}")
    folder_id = m.group(1)
    errors = []

    # ── Attempt 1: direct HTTP fetch + HTML parse ────────────────────────────
    try:
        folder_html_url = f"https://drive.google.com/drive/folders/{folder_id}"
        req = urllib.request.Request(
            folder_html_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        result = _parse_drive_html(html)
        if result:
            return result
        errors.append("HTTP scrape: folder page loaded but no video files found in HTML")
    except Exception as e:
        errors.append(f"HTTP scrape: {e}")

    # ── Attempt 2: gdown skip_download (no remaining_ok — removed in 5.x) ───
    try:
        import gdown as _gdown
        files = _gdown.download_folder(url, skip_download=True, quiet=True)
        if files is not None:
            result = []
            for f in files:
                fid  = getattr(f, 'id', None)
                path = getattr(f, 'path', None) or getattr(f, 'local_path', None)
                name = os.path.basename(str(path)) if path else None
                if fid and name and name.lower().endswith(_VIDEO_EXTS):
                    result.append({"id": str(fid), "name": name})
            if result:
                return result
            errors.append(
                f"gdown skip_download: found {len(files)} file(s) but none matched video extensions"
                if files else "gdown skip_download: folder appears empty"
            )
        else:
            errors.append("gdown skip_download: returned None — folder may be private or rate-limited")
    except Exception as e:
        errors.append(f"gdown skip_download: {e}")

    raise RuntimeError(
        "Could not list folder contents — all methods failed.\n"
        + "\n".join(f"  • {e}" for e in errors)
        + "\nMake sure the folder is shared as 'Anyone with the link'."
    )


def _parse_drive_html(html):
    """
    Extract video file IDs and names from a Google Drive folder HTML page.
    Google embeds file metadata in several JSON-like patterns; we try them all.
    Returns list of {'id': str, 'name': str} or empty list.
    """
    seen = set()
    result = []

    def _add(fid, fname):
        fname = fname.strip()
        key = (fid, fname)
        if key not in seen and fname.lower().endswith(_VIDEO_EXTS):
            seen.add(key)
            result.append({"id": fid, "name": fname})

    fid_re = r'[a-zA-Z0-9_-]{25,}'
    # filename: at least one char, a dot, 2-5 char extension, no quotes/backslashes
    fname_re = r'[^"\'\\]{1,200}\.[a-zA-Z0-9]{2,5}'

    # Pattern A: ["FILE_ID","FILENAME",...] — array literal in data blob
    for fid, fname in re.findall(rf'\["({fid_re})",\s*"({fname_re})"', html):
        _add(fid, fname)

    # Pattern B: "id":"FILE_ID","name":"FILENAME" — JSON object keys
    for fid, fname in re.findall(
        rf'"id"\s*:\s*"({fid_re})"[^}}]{{0,200}}"name"\s*:\s*"({fname_re})"', html
    ):
        _add(fid, fname)

    # Pattern C: data-id="FILE_ID" title="FILENAME" — HTML attributes
    for fid, fname in re.findall(
        rf'data-id="({fid_re})"[^>]{{0,200}}title="({fname_re})"', html
    ):
        _add(fid, fname)

    # Pattern D: reversed — name before id in JSON
    for fname, fid in re.findall(
        rf'"name"\s*:\s*"({fname_re})"[^}}]{{0,200}}"id"\s*:\s*"({fid_re})"', html
    ):
        _add(fid, fname)

    # Pattern E: Google's JS data arrays — ["ID",["mimeType",...],["NAME",...]]
    for fid, fname in re.findall(
        rf'"({fid_re})"[^]{{0,300}}"([^"\'\\]{{1,200}}\.[a-zA-Z0-9]{{2,5}})"', html
    ):
        _add(fid, fname)

    # Pattern F: plain filename near an ID anywhere on the same line
    for line in html.splitlines():
        fids = re.findall(rf'\b({fid_re})\b', line)
        fnames = re.findall(rf'([^\s"\'<>\\]{{1,120}}\.[a-zA-Z0-9]{{2,5}})', line)
        if len(fids) == 1 and len(fnames) == 1:
            _add(fids[0], fnames[0])

    return result


def _flatten_gdrive_tree(node):
    """
    Recursively flatten a gdown directory-structure result into video file dicts.
    Handles dicts, lists, and gdown's custom file-info objects (gdown 5.x).
    """
    result = []

    def _attr(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    def _walk(n):
        if n is None:
            return
        if isinstance(n, (list, tuple)):
            for item in n:
                _walk(item)
            return
        is_folder = _attr(n, "is_folder") or _attr(n, "type") == "folder"
        fid  = _attr(n, "id")
        name = _attr(n, "name")
        if not is_folder and fid and name:
            if str(name).lower().endswith(_VIDEO_EXTS):
                result.append({"id": str(fid), "name": str(name)})
        children = _attr(n, "children") or []
        for child in (children if isinstance(children, (list, tuple)) else [children]):
            _walk(child)

    _walk(node)
    return result


def _clean_drive_title(filename):
    """Produce a clean episode title from a raw Drive filename."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r'[_\-]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Normalize "ep998", "EP 998", "Ep998" → "Ep 998 ..."
    m = re.match(r'^[Ee][Pp]\s*(\d+)(.*)', name)
    if m:
        rest = m.group(2).strip()
        return f"Ep {int(m.group(1))}" + (f" {rest}" if rest else "")
    return name


def _ingest_one(job, video_path, filename, msg_q=None):
    """Ingest one downloaded file; logs to job record and optionally to msg_q."""
    episode_id = os.path.splitext(os.path.basename(filename))[0]
    if _already_ingested(episode_id):
        _job_log(job, "skipped", f"Skipped (already in DB): {filename}", msg_q)
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass
        return

    # ── Per-file record ───────────────────────────────────────────────────────
    file_rec = {"name": filename, "status": "transcribing",
                "started_at": datetime.now().isoformat(), "finished_at": None, "segments": 0}
    with _jobs_lock:
        job["files"].append(file_rec)
    # ──────────────────────────────────────────────────────────────────────────

    if not OPENAI_KEY:
        _job_log(job, "error",
                 f"{filename}: OPENAI_API_KEY not configured on server — cannot transcribe", msg_q)
        return

    title      = _clean_drive_title(filename)
    audio_path = video_path.rsplit(".", 1)[0] + "_audio.mp3"
    _job_log(job, "log", f"Transcribing {filename}…", msg_q)
    try:
        ffmpeg_ok = extract_audio(video_path, audio_path)
        result    = whisper_transcribe_chunked(audio_path) if ffmpeg_ok else whisper_transcribe(video_path)
        conn  = get_db()
        count = ingest_video_episode(conn, episode_id, title, video_path, result)
        _rebuild_fts(conn)
        conn.commit()
        conn.close()
        with _jobs_lock:
            file_rec["status"]      = "done"
            file_rec["finished_at"] = datetime.now().isoformat()
            file_rec["segments"]    = count
        _job_log(job, "done_file", f"Done: {filename} ({count} segments)", msg_q)
    except Exception as e:
        with _jobs_lock:
            file_rec["status"]      = "error"
            file_rec["finished_at"] = datetime.now().isoformat()
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass
        _job_log(job, "error", f"{filename}: {e}", msg_q)
    finally:
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass


def _drive_download_one(job, gdown, file_id_or_url, tmp_dir, msg_q):
    """
    Download a single Drive file into tmp_dir.
    Returns (local_path, stop_all). Retries with exponential backoff on 429.
    """
    free = _disk_free_bytes()
    if free < _MIN_FREE_BYTES:
        _job_log(job, "error",
                 f"STOPPING — only {_fmt_bytes(free)} free. "
                 f"Need at least {_fmt_bytes(_MIN_FREE_BYTES)} before downloading next file.",
                 msg_q)
        return None, True

    url = (
        f"https://drive.google.com/uc?id={file_id_or_url}"
        if not file_id_or_url.startswith("http")
        else file_id_or_url
    )

    _RATE_LIMIT_SIGNALS = ("429", "too many requests", "quota", "rate limit", "rate_limit")
    max_retries = 4
    base_delay  = 15  # seconds; doubles each attempt: 15, 30, 60, 120

    for attempt in range(max_retries):
        try:
            out = gdown.download(url, tmp_dir + os.sep, quiet=True)
        except Exception as e:
            err_lower = str(e).lower()
            is_rate   = any(s in err_lower for s in _RATE_LIMIT_SIGNALS)
            if is_rate and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                _job_log(job, "warn",
                         f"Drive rate-limited (429) — waiting {delay}s "
                         f"(retry {attempt + 2}/{max_retries})…", msg_q)
                time.sleep(delay)
                continue
            _job_log(job, "error",
                     f"Download failed: {e} — make sure link is 'Anyone with the link'", msg_q)
            return None, False

        if out and os.path.isfile(out):
            return out, False

        # gdown returned None — Drive sometimes does this silently on 429
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            _job_log(job, "warn",
                     f"Download returned no file (possible rate limit) — waiting {delay}s "
                     f"(retry {attempt + 2}/{max_retries})…", msg_q)
            time.sleep(delay)
        else:
            _job_log(job, "error",
                     f"Download failed after {max_retries} attempts — check share link is public", msg_q)

    return None, False


def _drive_worker(job, urls, msg_q=None):
    """
    Background worker: download from Google Drive + ingest, one file at a time.
    For folder URLs: lists contents first (no bulk download), then processes each
    file individually so disk never accumulates more than one video at once.
    """
    try:
        import gdown
    except ImportError:
        _job_log(job, "error", "gdown not installed on server (add to requirements.txt)", msg_q)
        _finish_job(job, "error")
        if msg_q: msg_q.put(None)
        return

    vdir     = videos_dir()
    tmp_base = temp_uploads_dir()
    os.makedirs(vdir,     exist_ok=True)
    os.makedirs(tmp_base, exist_ok=True)

    for i, url in enumerate(urls):
        is_folder = "/folders/" in url
        try:
            if is_folder:
                # ── FOLDER: list first, then download one at a time ──────────
                _job_log(job, "log", "Reading Drive folder contents (no download yet)…", msg_q)
                try:
                    folder_files = _list_gdrive_folder(url)
                except RuntimeError as e:
                    _job_log(job, "error", str(e), msg_q)
                    continue

                # Partition into new vs already ingested
                new_files = []
                for f in folder_files:
                    safe = secure_filename(f["name"])
                    eid  = os.path.splitext(safe)[0]
                    if _already_ingested(eid):
                        _job_log(job, "skipped", f"Skipped (already in DB): {f['name']}", msg_q)
                    else:
                        new_files.append(f)

                skip_count = len(folder_files) - len(new_files)
                _job_log(job, "log",
                         f"Folder has {len(folder_files)} video(s) — "
                         f"{len(new_files)} to import, {skip_count} already done",
                         msg_q)

                for idx, f in enumerate(new_files):
                    safe    = secure_filename(f["name"])
                    tmp_dir = os.path.join(tmp_base, f"gd_f_{int(time.time())}_{idx}")
                    os.makedirs(tmp_dir, exist_ok=True)

                    if idx > 0:
                        time.sleep(5)  # throttle between Drive downloads to avoid 429s

                    _job_log(job, "log",
                             f"[{idx+1}/{len(new_files)}] Downloading {f['name']} "
                             f"({_fmt_bytes(_disk_free_bytes())} free)…", msg_q)

                    out, stop = _drive_download_one(job, gdown, f["id"], tmp_dir, msg_q)
                    if stop:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        break   # disk full — stop this folder
                    if not out:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        continue

                    dst = os.path.join(vdir, safe)
                    shutil.move(out, dst)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    # Transcribe + ingest; temp audio is cleaned up inside _ingest_one
                    _ingest_one(job, dst, safe, msg_q)

            else:
                # ── SINGLE FILE ───────────────────────────────────────────────
                safe    = None  # unknown until after download
                tmp_dir = os.path.join(tmp_base, f"gd_file_{int(time.time())}_{i}")
                os.makedirs(tmp_dir, exist_ok=True)

                _job_log(job, "log",
                         f"Downloading file {i+1}/{len(urls)} "
                         f"({_fmt_bytes(_disk_free_bytes())} free)…", msg_q)

                out, stop = _drive_download_one(job, gdown, url, tmp_dir, msg_q)
                if stop or not out:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    if stop: break
                    continue

                fn   = os.path.basename(out)
                safe = secure_filename(fn)
                dst  = os.path.join(vdir, safe)
                shutil.move(out, dst)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                _ingest_one(job, dst, safe, msg_q)

        except Exception as e:
            _job_log(job, "error", str(e), msg_q)

    _finish_job(job, "done" if job["stats"]["errors"] == 0 else "error")
    if msg_q:
        msg_q.put(None)


@app.route("/api/drive-import")
@requires_auth
def drive_import():
    """SSE endpoint: downloads Google Drive share links and ingests each as a video episode."""
    urls_raw = request.args.get("urls", "")
    urls = [u.strip() for u in urls_raw.strip().splitlines() if u.strip()]
    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    msg_q = queue_module.Queue()
    job   = _new_job(urls)
    threading.Thread(target=_drive_worker, args=(job, urls, msg_q), daemon=True).start()

    def generate():
        yield f"data: {json.dumps({'type': 'log', 'msg': 'Connecting to Google Drive…'})}\n\n"
        while True:
            try:
                item = msg_q.get(timeout=20)
                if item is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except queue_module.Empty:
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )


@app.route("/api/video/<path:filename>")
@requires_auth
def serve_video(filename):
    vdir = videos_dir()
    file_path = os.path.join(vdir, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Not found"}), 404
    return send_file(file_path, mimetype="video/mp4", conditional=True)


def _transcripts_dir():
    env = os.environ.get("TRANSCRIPTS_DIR", "")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")


def _rebuild_fts(conn):
    """Rebuild FTS5 index from segments, preserving rowid alignment."""
    cur = conn.cursor()
    cur.execute("INSERT INTO fts_segments(fts_segments) VALUES('delete-all')")
    for row in conn.execute("SELECT rowid, episode_id, text FROM segments").fetchall():
        cur.execute(
            "INSERT INTO fts_segments(rowid, episode_id, text) VALUES (?, ?, ?)",
            (row[0], row[1], row[2]),
        )


@app.route("/api/delete-episode/<episode_id>", methods=["DELETE"])
@requires_auth
def delete_episode_endpoint(episode_id):
    conn = get_db()
    if not conn:
        return jsonify({"error": "No database"}), 500
    cur = conn.cursor()
    ep = conn.execute("SELECT video_path, filename FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    cur.execute("DELETE FROM segments WHERE episode_id = ?", (episode_id,))
    cur.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
    # Tombstone prevents ingest.py from re-adding this episode on next deploy
    cur.execute(
        "INSERT OR REPLACE INTO deleted_episodes (id, deleted_at) VALUES (?, ?)",
        (episode_id, datetime.now().isoformat())
    )
    _rebuild_fts(conn)
    conn.commit()
    conn.close()

    # Delete video file
    if ep and ep["video_path"] and os.path.exists(ep["video_path"]):
        try:
            os.remove(ep["video_path"])
        except Exception:
            pass

    return jsonify({"ok": True})


@app.route("/api/random-quote")
@requires_auth
def random_quote():
    """Return a random transcript segment to display as a rotating quote."""
    conn = get_db()
    if not conn:
        return jsonify({"quote": "", "episode": ""}), 200
    try:
        row = conn.execute(
            "SELECT s.text, e.title FROM segments s "
            "JOIN episodes e ON s.episode_id = e.id "
            "WHERE length(s.text) > 80 AND length(s.text) < 400 "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return jsonify({"quote": row["text"].strip(), "episode": row["title"]})
        return jsonify({"quote": "", "episode": ""})
    except Exception:
        return jsonify({"quote": "", "episode": ""})


@app.route("/api/rename-episode/<episode_id>", methods=["POST"])
@requires_auth
def rename_episode(episode_id):
    """Rename an episode's title."""
    data = request.get_json() or {}
    new_title = (data.get("title") or "").strip()
    if not new_title:
        return jsonify({"error": "Title required"}), 400
    conn = get_db()
    if not conn:
        return jsonify({"error": "No database"}), 500
    conn.execute("UPDATE episodes SET title = ? WHERE id = ?", (new_title, episode_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/purge-no-video", methods=["DELETE"])
@requires_auth
def purge_no_video():
    """Delete all episodes (and their segments) that have no video file on disk."""
    conn = get_db()
    if not conn:
        return jsonify({"error": "No database"}), 500

    rows = conn.execute("SELECT id, video_path FROM episodes").fetchall()
    to_delete = [
        r["id"] for r in rows
        if not r["video_path"] or not os.path.exists(r["video_path"])
    ]

    if not to_delete:
        conn.close()
        return jsonify({"deleted": 0, "ids": []})

    now = datetime.now().isoformat()
    cur = conn.cursor()
    for ep_id in to_delete:
        cur.execute("DELETE FROM segments WHERE episode_id = ?", (ep_id,))
        cur.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
        # Tombstone prevents ingest.py from re-adding on next deploy
        cur.execute(
            "INSERT OR REPLACE INTO deleted_episodes (id, deleted_at) VALUES (?, ?)",
            (ep_id, now)
        )
    _rebuild_fts(conn)
    conn.commit()
    conn.close()
    return jsonify({"deleted": len(to_delete), "ids": to_delete})


@app.route("/api/clip/<path:filename>")
@requires_auth
def download_clip(filename):
    """Extract and download a clip from a video using ffmpeg."""
    start = float(request.args.get("start", 0))
    duration = float(request.args.get("duration", 90))
    duration = min(duration, 600)  # cap at 10 minutes

    vdir = videos_dir()
    video_path = os.path.join(vdir, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    import tempfile
    clip_name = f"clip_{int(start)}s.mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    preview = request.args.get("preview") == "1"

    try:
        if preview:
            # Small 720p preview blob for the clip viewer (~8-15 MB, instant seeking)
            cmd = [
                "ffmpeg", "-threads", "0",
                "-ss", str(start), "-i", video_path, "-t", str(duration),
                "-vf", "scale=1280:-2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                "-y", tmp_path
            ]
            timeout = 300
        else:
            # Full-quality copy for the final Save
            cmd = [
                "ffmpeg",
                "-ss", str(start), "-i", video_path, "-t", str(duration),
                "-c", "copy", "-movflags", "+faststart",
                "-y", tmp_path
            ]
            timeout = 120

        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            return jsonify({"error": "Clip extraction failed"}), 500

        @after_this_request
        def cleanup(response):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return response

        # Serve inline (no attachment header) so <video src=""> works natively.
        # Clients that want a file download use fetch() + saveVideoBlob() on the client side.
        return send_file(tmp_path, mimetype="video/mp4")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500


@app.route("/api/episode-segments/<episode_id>")
@requires_auth
def episode_segments_by_id(episode_id):
    """Return all segments for a specific episode by its TEXT id (used for video episodes)."""
    conn = get_db()
    if not conn:
        return jsonify({"results": []})
    rows = conn.execute("""
        SELECT s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text,
               e.title AS episode_title, e.filename, e.video_path, 0 AS rank
        FROM segments s JOIN episodes e ON e.id = s.episode_id
        WHERE e.id = ?
        ORDER BY s.start_secs ASC
    """, (episode_id,)).fetchall()
    conn.close()
    return jsonify({"results": [format_result(r, 0) for r in rows]})


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/dev")
def dev_dashboard():
    # No @requires_auth — the page renders its own login form when unauthenticated
    return send_from_directory("static", "dev.html")


@app.route("/api/dev/status")
@requires_auth
def dev_status():
    """Return system stats + all tracked import jobs."""
    # DB stats
    conn = get_db()
    ep_count  = 0
    seg_count = 0
    vid_count = 0
    if conn:
        ep_count  = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        seg_count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        vid_count = conn.execute("SELECT COUNT(*) FROM episodes WHERE video_path IS NOT NULL AND video_path != ''").fetchone()[0]
        conn.close()

    # Disk stats
    db_bytes    = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    vdir        = videos_dir()
    videos_bytes = 0
    video_count_files = 0
    if os.path.isdir(vdir):
        for f in os.listdir(vdir):
            fp = os.path.join(vdir, f)
            if os.path.isfile(fp):
                videos_bytes += os.path.getsize(fp)
                video_count_files += 1

    disk_total = disk_free = 0
    try:
        du = shutil.disk_usage("/data" if os.path.isdir("/data") else os.path.dirname(DB_PATH))
        disk_total = du.total
        disk_free  = du.free
    except Exception:
        pass

    with _jobs_lock:
        jobs_snapshot = [
            {
                "id":          j["id"],
                "started_at":  j["started_at"],
                "finished_at": j["finished_at"],
                "status":      j["status"],
                "urls":        j["urls"],
                "stats":       dict(j["stats"]),
                "logs":        list(j["logs"]),
                "files":       list(j.get("files", [])),
            }
            for j in _jobs
        ]

    # Search history — last 200 queries, newest first
    searches = []
    try:
        conn = get_db()
        if conn:
            rows = conn.execute(
                "SELECT query, ep_filter, results, searched_at FROM search_history "
                "ORDER BY id DESC LIMIT 200"
            ).fetchall()
            conn.close()
            searches = [dict(r) for r in rows]
    except Exception:
        pass

    return jsonify({
        "episodes":            ep_count,
        "segments":            seg_count,
        "video_episodes":      vid_count,
        "transcript_episodes": ep_count - vid_count,
        "db_bytes":            db_bytes,
        "videos_bytes":        videos_bytes,
        "video_files":         video_count_files,
        "disk_total":          disk_total,
        "disk_free":           disk_free,
        "jobs":                jobs_snapshot,
        "searches":            searches,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)