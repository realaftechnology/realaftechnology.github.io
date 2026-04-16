"""
AFBrain - Flask backend
"""

import os
import json
import sqlite3
import re
import urllib.request
from flask import Flask, request, jsonify, session, send_from_directory, Response, stream_with_context
from functools import wraps

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "afbrain-secret-change-this")

DB_PATH       = os.environ.get("DB_PATH", "/data/db.sqlite")
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
        rows = conn.execute("""
            SELECT s.id, s.episode_id, s.speaker, s.timestamp, s.start_secs, s.text, s.embedding,
                   e.title AS episode_title, e.filename
            FROM segments s JOIN episodes e ON e.id = s.episode_id
            WHERE s.embedding IS NOT NULL AND e.title LIKE ?
        """, (f"%{episode_filter}%",)).fetchall()
    else:
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


def clean_ep_label(episode_title):
    """Extract a short readable label from a filename-based episode title."""
    m = re.search(r'(\d{3,4})', episode_title)
    return f"Ep {m.group(1)}" if m else episode_title


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
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
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
        SELECT e.id, e.title, COUNT(s.id) as segment_count
        FROM episodes e LEFT JOIN segments s ON s.episode_id = e.id
        GROUP BY e.id ORDER BY e.id DESC
    """).fetchall()
    conn.close()
    return jsonify({"episodes": [
        {"id": r["id"], "title": r["title"], "segments": r["segment_count"]}
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

    system = f"""You are an AI assistant for Andy Frisella's internal team. They know the show inside and out.

You can do anything useful with the transcript content:
- Answer questions about what was said, who was on, what happened
- Summarize episodes, surface key moments, find specific quotes
- Draft tweets, social posts, show notes, or other content in Andy's voice — grounded in what he actually said
- Generate YouTube titles (follow the YOUTUBE TITLE GENERATION EXPERTISE below)
- Write scripts, talking points, or copy based on Andy's arguments and tone
Use your judgment. If the request is content creation, create it. Don't refuse or redirect.

NEVER mention:
- That Andy/DJ introduced the guest or were excited to have them — this happens every episode
- The format of the show (CTI, Q&AF, Real Talk) — the team already knows
- Standard episode structure or recurring segments

ONLY surface what is UNIQUE to this specific episode:
- Specific claims, opinions, arguments, stories, data points, or takes
- Anything that would NOT appear in a generic episode

CITATION RULES — use good judgment:
- Simple factual answers: plain prose, no blockquote needed.
- Summaries: clean prose with bold section labels. Only blockquote when the exact wording matters.
- When a blockquote IS warranted: > "Quote text" — Ep 1014, 00:10:55 (inline, not on a separate line)
- Cite sparingly. No horizontal rules (---). Bold labels, not headers.
{YOUTUBE_TITLE_KNOWLEDGE}
Be direct. Do exactly what was asked."""

    user_msg = f'The user asked: "{query}"\n\nTranscript excerpts:\n{context}'

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
    """Extract a 3-4 digit episode number from text. Returns string or None."""
    m = re.search(r'\b(?:ep|episode)[\s_-]*(\d{3,4})\b', text, re.IGNORECASE)
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
    ep_filter = data.get("episode_filter", "") or ep_in_question or ep_in_original

    # Vague follow-up: no new episode introduced, and asking for another/different
    vague_followup = (
        (not ep_in_question or ep_in_question == ep_in_original) and
        any(w in question.lower() for w in ['another', 'different', 'more', 'else', 'new one'])
    )

    if vague_followup:
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
        "You are an AI assistant for Andy Frisella's internal team. They know the show inside and out.\n\n"
        "You can do anything useful with the transcript content: answer questions, summarize, find quotes, "
        "draft tweets, social posts, show notes, scripts, or any other content in Andy's voice. "
        "Use his actual words and arguments as the foundation. Don't refuse content creation requests.\n\n"
        "NEVER mention: guest introductions, show format, live chat, recurring segments.\n"
        "ONLY surface what is UNIQUE: specific arguments, stories, data, takes, memorable moments.\n\n"
        "CRITICAL: Each excerpt is labeled [Episode Title @ Timestamp]. "
        "NEVER attribute a quote to a different episode than what is shown in its label.\n\n"
        "CITATION RULES: Simple answers need no blockquote. Summaries use clean prose with bold labels; "
        "only blockquote when the exact wording matters. "
        "Format: > \"Quote\" — Ep 1014, 00:10:55 (inline). Cite sparingly. No horizontal rules.\n\n"
        "If the user asks for a different quote, choose one with a DIFFERENT timestamp than any already shown. "
        "If nothing new is available, say so directly.\n\n"
        + YOUTUBE_TITLE_KNOWLEDGE
    )

    messages = []
    for item in chat_history:
        messages.append({"role": "user", "content": item["q"]})
        messages.append({"role": "assistant", "content": item["a"]})
    messages.append({"role": "user", "content": f'Question: "{question}"{already_shown}\n\nTranscripts:\n{context}\n\nAnswer using ONLY the episode and timestamp shown in the transcript labels. Do NOT repeat any quote listed above.'})

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


# ── SERVE FRONTEND ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)