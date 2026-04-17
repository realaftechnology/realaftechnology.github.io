"""
AFBrain - Flask backend
"""

import os
import json
import sqlite3
import re
import subprocess
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, session, send_from_directory, send_file, Response, stream_with_context
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "afbrain-secret-change-this")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

DB_PATH       = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "db.sqlite"))
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")   # used only for embeddings
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "") # used for all AI text generation

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
    # Ensure video_path column exists (migration for older dbs)
    try:
        conn.execute("ALTER TABLE episodes ADD COLUMN video_path TEXT")
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
        # Titles look like "Ep 01 - ..." or "Ep 979 - ..."
        # Match number followed by space, dash, or dot — not another digit
        patterns = [f"Ep {n} %", f"Ep {n}-%", f"Ep {n}.%"]
        if n < 100:
            p = f"{n:02d}"  # zero-padded: 1 -> 01
            patterns += [f"Ep {p} %", f"Ep {p}-%", f"Ep {p}.%"]
        sql = "(" + " OR ".join(["e.title LIKE ?"] * len(patterns)) + ")"
        return sql, patterns
    except ValueError:
        return "e.title LIKE ?", [f"%{ep_number}%"]


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
        SELECT e.id, e.title, e.video_path, COUNT(s.id) as segment_count
        FROM episodes e LEFT JOIN segments s ON s.episode_id = e.id
        GROUP BY e.id ORDER BY e.id DESC
    """).fetchall()
    conn.close()
    return jsonify({"episodes": [
        {"id": r["id"], "title": r["title"], "segments": r["segment_count"],
         "has_video": bool(r["video_path"]),
         "video_filename": os.path.basename(r["video_path"]) if r["video_path"] else None}
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
            max_tokens=2000,
            temperature=0.4,
        )),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def extract_episode_filter(text):
    """Extract an episode number from text. Returns string or None."""
    m = re.search(r'\b(?:ep|episode)[\s_-]*(\d+)\b', text, re.IGNORECASE)
    if m: return m.group(1)
    m = re.search(r'\b(\d{4})\b', text)
    if m: return m.group(1)
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
            max_tokens=1500,
            temperature=0.4,
        )),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── VIDEO ─────────────────────────────────────────────────────────────────────

def videos_dir():
    return os.path.join(os.path.dirname(DB_PATH), "videos")


def secs_to_timestamp(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def extract_audio(video_path, audio_path):
    """Extract audio from video to mp3 using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-ab", "64k", "-y", audio_path],
        capture_output=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")


def whisper_transcribe(audio_path):
    """Transcribe audio via OpenAI Whisper API. Returns verbose_json dict."""
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


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
        "INSERT INTO episodes (id, title, filename, video_path) VALUES (?, ?, ?, ?)",
        (episode_id, title, os.path.basename(video_path), video_path)
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
        extract_audio(video_path, audio_path)
        result = whisper_transcribe(audio_path)
        episode_id = filename.rsplit(".", 1)[0]
        title = request.form.get("title", "").strip() or episode_id
        conn = get_db()
        count = ingest_video_episode(conn, episode_id, title, video_path, result)
        # Rebuild FTS
        cur = conn.cursor()
        cur.execute("INSERT INTO fts_segments(fts_segments) VALUES('delete-all')")
        for row in conn.execute("SELECT episode_id, text FROM segments").fetchall():
            cur.execute("INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)", row)
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


@app.route("/api/video/<path:filename>")
@requires_auth
def serve_video(filename):
    vdir = videos_dir()
    file_path = os.path.join(vdir, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Not found"}), 404
    return send_file(file_path, mimetype="video/mp4", conditional=True)


@app.route("/api/delete-episode/<episode_id>", methods=["DELETE"])
@requires_auth
def delete_episode_endpoint(episode_id):
    conn = get_db()
    if not conn:
        return jsonify({"error": "No database"}), 500
    cur = conn.cursor()
    ep = conn.execute("SELECT video_path FROM episodes WHERE id = ?", (episode_id,)).fetchone()
    cur.execute("DELETE FROM segments WHERE episode_id = ?", (episode_id,))
    cur.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
    cur.execute("INSERT INTO fts_segments(fts_segments) VALUES('delete-all')")
    for row in conn.execute("SELECT episode_id, text FROM segments").fetchall():
        cur.execute("INSERT INTO fts_segments (episode_id, text) VALUES (?, ?)", row)
    conn.commit()
    conn.close()
    if ep and ep["video_path"] and os.path.exists(ep["video_path"]):
        try:
            os.remove(ep["video_path"])
        except Exception:
            pass
    return jsonify({"ok": True})


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)