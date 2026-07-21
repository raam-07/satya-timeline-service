import os
import sys
import json
import time
import logging
import re
import zlib
import sqlite3
import argparse
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load env variables from .env if present
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env()

# Self-healing default local path for DB
default_db_path = '/Users/mac/Downloads/Code/Satya/satya.db'
if not os.path.exists(os.path.dirname(default_db_path)):
    default_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'satya.db')

DB_PATH = os.environ.get('SATYA_DB_PATH', default_db_path)

# --- Shard-replay globals (empty / 0 = normal mode) ---
SHARD_CTX = {}
DEADLINE_TS = 0

# Hard cap on saga-check LLM calls per event. Articles with ubiquitous entity
# keys (bjp, nda, big states) can match hundreds of closed events; unbounded,
# that loop runs for hours (cause of the 2026-07-10 6h-timeout kill).
MAX_SAGA_LLM_CALLS = 8

class DeadlineReached(Exception):
    """Raised inside long LLM loops when DEADLINE_TS passes; callers treat the
    interrupted unit of work as not-done and it is retried on the next run."""
    pass

# Article eligibility filter — single source of truth (export_snapshot.py imports this).
ELIGIBILITY_SQL = """a.status IN ('classified','entity_processed','processed')
              AND (a.category != 'international'
                   OR a.party_mentioned NOT IN ('[]','')
                   OR a.ministers_mentioned NOT IN ('[]','')
                   OR a.states_mentioned NOT IN ('[]','')
                   OR a.cities_mentioned NOT IN ('[]',''))
              AND (a.ministers_mentioned != '[]' OR a.party_mentioned != '[]' OR a.civic_flag = 1)"""

# --- Tracked figures (simple, local, zero-dependency) ---
# tracked_figures.txt holds names registered via the Timeline Doctor's name
# mode (or by hand). The forward runner treats a title match as eligible and
# stamps the figure's key on the article — so one Doctor run keeps a person
# tracked forever, without touching the classifier or entity library.
def load_tracked_figures():
    path = os.path.join(os.path.dirname(__file__), 'tracked_figures.txt')
    names = []
    try:
        with open(path) as f:
            for line in f:
                n = line.strip()
                if len(n) >= 5 and not n.startswith('#'):
                    names.append(n)
    except FileNotFoundError:
        pass
    return names

TRACKED_FIGURES = load_tracked_figures()

def tracked_match_sql():
    """(clause, params): does the article's title mention a tracked figure?
    Returns a never-true clause when the list is empty (zero behavior change)."""
    if not TRACKED_FIGURES:
        return "0", []
    parts, params = [], []
    for n in TRACKED_FIGURES:
        like = "%" + n.replace("%", "").replace("_", " ") + "%"
        parts.append("(a.title LIKE ? OR a.rephrased_title LIKE ?)")
        params += [like, like]
    return "(a.status IN ('classified','entity_processed','processed') AND (" + " OR ".join(parts) + "))", params

def tracked_keys_for(text):
    keys = []
    low = (text or '').lower()
    for n in TRACKED_FIGURES:
        if n.lower() in low:
            k = re.sub(r'[^a-z0-9]+', '_', n.lower()).strip('_')
            if k:
                keys.append(k)
    return keys

# Local schema for a shard's own event DB (mirrors production tables, plus
# last_scraped_at on the checkpoint because shards process in (scraped_at, id)
# order, and shard_meta for the done flag).
SHARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    slug TEXT,
    entity_keys TEXT,
    centroid BLOB,
    first_seen INTEGER,
    last_seen INTEGER,
    article_count INTEGER,
    state TEXT,
    scope TEXT,
    saga_id INTEGER
);
CREATE TABLE IF NOT EXISTS event_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    article_id INTEGER,
    milestone TEXT,
    event_date INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ea_event ON event_articles(event_id);
CREATE INDEX IF NOT EXISTS idx_ea_article ON event_articles(article_id);
CREATE TABLE IF NOT EXISTS timeline_checkpoint (
    id INTEGER PRIMARY KEY,
    last_article_id INTEGER,
    last_scraped_at INTEGER
);
CREATE TABLE IF NOT EXISTS shard_meta (
    id INTEGER PRIMARY KEY,
    shard_done INTEGER DEFAULT 0
);
"""

def get_db_connection():
    # Shard-replay mode: everything is local. The shard's own event DB is the
    # main database; the read-only articles snapshot is ATTACHed as 'snap'.
    # Unqualified 'articles' references resolve to snap.articles because main
    # has no articles table. Turso is NEVER touched in this mode.
    if SHARD_CTX:
        conn = sqlite3.connect(SHARD_CTX['shard_db'])
        conn.executescript(SHARD_SCHEMA)
        conn.commit()
        conn.execute("ATTACH DATABASE ? AS snap", (SHARD_CTX['snapshot_db'],))
        logging.info(f"[SHARD {SHARD_CTX['shard']}] Local shard DB: {SHARD_CTX['shard_db']} | snapshot attached: {SHARD_CTX['snapshot_db']}")
        return conn

    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            logging.info(f"Connecting to remote Turso Database at: {db_url}")
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local SQLite.")
            
    logging.info(f"Connecting to local SQLite Database at: {DB_PATH}")
    return sqlite3.connect(DB_PATH)

# Decompress rephrased_article zlib BLOB helper
def decompress_text(blob):
    if not blob:
        return ""
    try:
        return zlib.decompress(blob).decode('utf-8')
    except Exception:
        try:
            return zlib.decompress(blob, 16 + zlib.MAX_WBITS).decode('utf-8')
        except Exception:
            try:
                return blob.decode('utf-8')
            except Exception:
                return str(blob)

# Entity key parsing
def extract_keys(party, minister, state, city):
    keys = set()
    for item in [party, minister, state, city]:
        if not item:
            continue
        try:
            lst = json.loads(item) if isinstance(item, str) else item
            if isinstance(lst, list):
                for x in lst:
                    clean = x.lower().strip().replace(" ", "_").replace(".", "")
                    if clean:
                        keys.add(clean)
        except Exception:
            pass
    return list(keys)

def get_founding_milestone(cursor, event_id):
    cursor.execute("""
        SELECT ea.milestone, a.rephrased_title, a.title
        FROM event_articles ea
        JOIN articles a ON ea.article_id = a.id
        WHERE ea.event_id = ?
        ORDER BY ea.event_date ASC LIMIT 1
    """, (event_id,))
    row = cursor.fetchone()
    if row:
        milestone, reph_title, orig_title = row
        if milestone and milestone.strip():
            return milestone.strip()
        if reph_title and reph_title.strip():
            return reph_title.strip()
        if orig_title and orig_title.strip():
            return orig_title.strip()
    return "Untitled Founding Article"

def assign_saga_id(cursor, ev1_id, ev2_id):
    cursor.execute("SELECT saga_id FROM events WHERE id = ?", (ev1_id,))
    row1 = cursor.fetchone()
    s1 = row1[0] if row1 else None
    
    cursor.execute("SELECT saga_id FROM events WHERE id = ?", (ev2_id,))
    row2 = cursor.fetchone()
    s2 = row2[0] if row2 else None
    
    if s1 is None and s2 is None:
        cursor.execute("SELECT COALESCE(MAX(saga_id), 0) + 1 FROM events")
        new_saga_id = cursor.fetchone()[0]
        cursor.execute("UPDATE events SET saga_id = ? WHERE id IN (?, ?)", (new_saga_id, ev1_id, ev2_id))
    elif s1 is not None and s2 is None:
        cursor.execute("UPDATE events SET saga_id = ? WHERE id = ?", (s1, ev2_id))
    elif s1 is None and s2 is not None:
        cursor.execute("UPDATE events SET saga_id = ? WHERE id = ?", (s2, ev1_id))
    elif s1 != s2:
        # Both have different saga_ids, unify (merge s2 into s1)
        cursor.execute("UPDATE events SET saga_id = ? WHERE saga_id = ?", (s1, s2))
        cursor.execute("UPDATE events SET saga_id = ? WHERE id = ?", (s1, ev2_id))

def generate_event_scope(llm_9b, template, title, summary):
    try:
        prompt = template.format(
            title=title or "Untitled Article",
            summary=summary[:1200] if summary else "No Summary Available"
        )
        output = llm_9b(prompt, max_tokens=150, stop=["<|im_end|>"], temperature=0.0)
        scope = output['choices'][0]['text'].strip()

        if not scope:
            return None
        # Qwen often numbers the two sentences ("1. This event covers... 2. It
        # excludes...") and sometimes wraps them in quotes. Normalize instead
        # of rejecting — the content is fine, only the decoration differs.
        lines = []
        for line in scope.splitlines():
            line = line.strip()
            line = re.sub(r'^\s*\d+[\.\)]\s*', '', line)  # strip "1." / "2)"
            line = line.strip('"“” ')            # strip quotes
            if line:
                lines.append(line)
        scope = ' '.join(lines).strip()

        words = scope.split()
        if len(words) > 90:
            logging.warning(f"Scope too long ({len(words)} words): '{scope}'")
            return None
        if not scope.lower().startswith("this event covers"):
            logging.warning(f"Scope does not start with 'This event covers': '{scope}'")
            return None
        return scope
    except Exception as e:
        logging.error(f"Error in generate_event_scope: {e}")
        return None

def check_saga_links_in_memory(cursor, llm_9b, event_id, event_title, merged_keys, first_seen, last_seen, saga_check_prompt_template):
    linked_ids = []
    t_keys = set(merged_keys)
    t_founding = get_founding_milestone(cursor, event_id)
    
    # Fetch target current saga_id and scope
    cursor.execute("SELECT saga_id, scope FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    t_saga = row[0] if row else None
    t_scope = row[1] if row else None
    
    # Query all closed events
    cursor.execute("SELECT id, title, entity_keys, first_seen, last_seen, saga_id, scope FROM events WHERE state = 'closed'")
    closed_events = cursor.fetchall()
    
    # Phase 1: collect candidates cheaply (no LLM), strongest overlap first.
    candidates = []
    for c_id, c_title, c_keys_json, c_first, c_last, c_saga, c_scope in closed_events:
        if c_id == event_id:
            continue
        if not c_title:
            continue
        # Skip if they already share the same saga_id
        if t_saga is not None and c_saga is not None and t_saga == c_saga:
            continue

        c_keys = set(json.loads(c_keys_json))
        overlap = len(t_keys.intersection(c_keys))
        if overlap >= 2:
            candidates.append((overlap, c_last or 0, c_id, c_title, c_first, c_last, c_saga, c_scope))

    # Phase 2: cap LLM calls — highest overlap wins, recency breaks ties.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    if len(candidates) > MAX_SAGA_LLM_CALLS:
        logging.info(f"[SAGA CHECK] Event {event_id}: {len(candidates)} candidates share >=2 keys; capped to {MAX_SAGA_LLM_CALLS} strongest.")
        candidates = candidates[:MAX_SAGA_LLM_CALLS]

    for idx, (overlap, _, c_id, c_title, c_first, c_last, c_saga, c_scope) in enumerate(candidates):
        if DEADLINE_TS and time.time() >= DEADLINE_TS:
            logging.info(f"[SAGA CHECK] Event {event_id}: deadline reached after {idx}/{len(candidates)} calls — returning partial links.")
            break
        c_founding = get_founding_milestone(cursor, c_id)

        # Determine scopes with fallbacks
        c_scope_val = c_scope or c_founding
        t_scope_val = t_scope or t_founding

        def fmt_date(ts):
            if not ts: return "N/A"
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

        if c_first < first_seen:
            ch_a = {"title": c_title, "scope": c_scope_val, "first": fmt_date(c_first), "last": fmt_date(c_last)}
            ch_b = {"title": event_title, "scope": t_scope_val, "first": fmt_date(first_seen), "last": fmt_date(last_seen)}
        else:
            ch_a = {"title": event_title, "scope": t_scope_val, "first": fmt_date(first_seen), "last": fmt_date(last_seen)}
            ch_b = {"title": c_title, "scope": c_scope_val, "first": fmt_date(c_first), "last": fmt_date(c_last)}

        prompt = saga_check_prompt_template.format(
            a_title=ch_a["title"],
            a_scope=ch_a["scope"],
            a_first_seen=ch_a["first"],
            a_last_seen=ch_a["last"],
            b_title=ch_b["title"],
            b_scope=ch_b["scope"],
            b_first_seen=ch_b["first"],
            b_last_seen=ch_b["last"]
        )

        logging.info(f"[SAGA CHECK] Event {event_id} vs closed Event {c_id} ({idx+1}/{len(candidates)}, {overlap} shared keys)...")
        output = llm_9b(prompt, max_tokens=100, stop=["<|im_end|>"], temperature=0.0)
        res_text = output['choices'][0]['text'].strip().upper()

        if res_text.startswith("SAME_SAGA"):
            logging.info(f"[SAGA LINK MEMORY] Event {event_id} and Event {c_id} linked as SAME_SAGA.")
            linked_ids.append(c_id)
            if t_saga is None:
                t_saga = c_saga
    return linked_ids

def evict_article_to_new_event(cursor, encoder, event_id, article_id):
    cursor.execute("""
        SELECT rephrased_title, title, rephrased_article, scraped_at,
               party_mentioned, ministers_mentioned, states_mentioned, cities_mentioned
        FROM articles WHERE id = ?
    """, (article_id,))
    row = cursor.fetchone()
    if not row:
        return
    
    reph_title, orig_title, reph_art_blob, scraped_at, party, minister, state, city = row
    title_text = reph_title or orig_title or ""
    article_text = decompress_text(reph_art_blob)
    combined_text = f"{title_text} {article_text}".strip()
    
    embedding = encoder.encode(combined_text, convert_to_numpy=True)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
        
    art_entities = extract_keys(party, minister, state, city)
    
    # Create new event (invisible, state='closed')
    cursor.execute("""
        INSERT INTO events (title, slug, entity_keys, centroid, first_seen, last_seen, article_count, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (None, None, json.dumps(art_entities), embedding.tobytes(), scraped_at, scraped_at, 1, 'closed'))
    new_ev_id = cursor.lastrowid
    
    # Move article to new event
    cursor.execute("""
        UPDATE event_articles
        SET event_id = ?
        WHERE event_id = ? AND article_id = ?
    """, (new_ev_id, event_id, article_id))

def parse_evict_line(res_text, valid_ids):
    """Extract evicted article IDs from the model's 'EVICT: ...' verdict line.
    Defensive: only IDs actually in valid_ids count; anything malformed or
    missing means NO evictions (a rare unscrubbed event beats a crashed run)."""
    try:
        verdict = None
        for line in res_text.splitlines():
            if 'EVICT' in line.upper():
                verdict = line  # keep last matching line (final answer)
        if verdict is None:
            return [], False
        after = verdict.upper().split('EVICT', 1)[1]
        if 'NONE' in after:
            return [], True
        found = [int(x) for x in re.findall(r'\d+', after)]
        hallucinated = [x for x in found if x not in valid_ids]
        if hallucinated:
            logging.warning(f"[BATCH AUDIT] Model returned unknown article IDs {hallucinated} — ignoring those.")
        return [x for x in found if x in valid_ids], True
    except Exception as e:
        logging.warning(f"[BATCH AUDIT] Failed to parse verdict line: {e}")
        return [], False

BATCH_AUDIT_CHUNK = 30  # milestones per audit call; big events use several calls

def get_closure_decisions(cursor, llm_9b, event_id, closure_audit_prompt_template, saga_check_prompt_template):
    evicted_article_ids = []
    linked_event_ids = []

    # 1. Exit Audit
    cursor.execute("SELECT COUNT(*) FROM event_articles WHERE event_id = ?", (event_id,))
    article_count = cursor.fetchone()[0]

    if article_count > 2:
        cursor.execute("SELECT title, first_seen, last_seen, entity_keys, scope FROM events WHERE id = ?", (event_id,))
        ev_row = cursor.fetchone()
        event_title = ev_row[0] or "Untitled Event"
        first_seen = ev_row[1] or 0
        last_seen = ev_row[2] or 0
        entity_keys = json.loads(ev_row[3]) if ev_row[3] else []
        event_scope = ev_row[4]
        
        founding_milestone = get_founding_milestone(cursor, event_id)
        scope_val = event_scope or founding_milestone
        
        cursor.execute("""
            SELECT ea.article_id, ea.milestone, a.title, a.rephrased_title
            FROM event_articles ea
            JOIN articles a ON ea.article_id = a.id
            WHERE ea.event_id = ?
        """, (event_id,))
        member_articles = cursor.fetchall()
        
        # Batch audit: judge ALL milestones in one call per chunk of 30,
        # instead of one 5-minute call per article. Same evidence (title +
        # milestone lines), two orders of magnitude fewer LLM calls.
        valid_ids = set()
        member_lines_all = []
        for art_id, milestone, orig_title, reph_title in member_articles:
            art_milestone = milestone or reph_title or orig_title or "No Title"
            valid_ids.add(int(art_id))
            member_lines_all.append(f"{art_id}: {art_milestone}")

        for i in range(0, len(member_lines_all), BATCH_AUDIT_CHUNK):
            if DEADLINE_TS and time.time() >= DEADLINE_TS:
                # Abort the whole ceremony: closing with partial evict decisions
                # would be wrong. Event stays open and is retried next run.
                raise DeadlineReached(f"Deadline during closure audit of event {event_id} — will retry next run.")
            chunk = member_lines_all[i:i + BATCH_AUDIT_CHUNK]
            prompt = closure_audit_prompt_template.format(
                event_title=event_title,
                scope=scope_val,
                count=len(chunk),
                member_lines="\n".join(chunk)
            )
            logging.info(f"[BATCH AUDIT] Event {event_id}: auditing articles {i+1}-{i+len(chunk)} of {len(member_lines_all)} in one call...")
            output = llm_9b(prompt, max_tokens=500, stop=["<|im_end|>"], temperature=0.0)
            res_text = output['choices'][0]['text'].strip()

            chunk_evicted, parsed_ok = parse_evict_line(res_text, valid_ids)
            if not parsed_ok:
                logging.warning(f"[BATCH AUDIT] Event {event_id}: no parsable verdict — keeping all articles in this chunk. Raw: {res_text[:200]}")
            for art_id in chunk_evicted:
                logging.info(f"[EVICT DECISION] Event {event_id}: Evict Article {art_id}")
                evicted_article_ids.append(art_id)
            kept = len(chunk) - len(chunk_evicted)
            logging.info(f"[BATCH AUDIT] Event {event_id}: chunk verdict — keep {kept}, evict {len(chunk_evicted)}.")
    else:
        cursor.execute("SELECT title, first_seen, last_seen, entity_keys FROM events WHERE id = ?", (event_id,))
        ev_row = cursor.fetchone()
        event_title = ev_row[0] if ev_row else "Untitled Event"
        first_seen = ev_row[1] if ev_row else 0
        last_seen = ev_row[2] if ev_row else 0
        entity_keys = json.loads(ev_row[3]) if ev_row and ev_row[3] else []
        
    # 2. Saga Link Check
    if event_title:
        linked_event_ids = check_saga_links_in_memory(
            cursor, llm_9b, event_id, event_title, entity_keys, first_seen, last_seen, saga_check_prompt_template
        )
        
    return evicted_article_ids, linked_event_ids

def reframe_event(cursor, llm_9b, event_id, title_tmpl, scope_tmpl, min_articles=5):
    """Re-derive title AND scope from the event's FULL milestone set. Fixes
    framing drift: an event founded from one article (or absorbing a bigger
    sub-story later) otherwise keeps its stale founding title/scope forever.
    Run at closure, when all articles are in. No-op for tiny events."""
    cursor.execute("SELECT COUNT(*) FROM event_articles WHERE event_id = ?", (event_id,))
    if cursor.fetchone()[0] < min_articles:
        return
    cursor.execute("""
        SELECT COALESCE(ea.milestone, a.rephrased_title, a.title)
        FROM event_articles ea JOIN articles a ON a.id = ea.article_id
        WHERE ea.event_id = ? ORDER BY ea.event_date ASC
    """, (event_id,))
    milestones = [r[0] for r in cursor.fetchall() if r[0]]
    if len(milestones) < min_articles:
        return
    try:
        # Title from the whole arc (sample across the story, not just the top)
        sample = milestones[:20]
        out = llm_9b(title_tmpl.format(milestones="\n".join(f"- {m}" for m in sample)),
                     max_tokens=60, stop=["<|im_end|>"], temperature=0.0)
        new_title = clean_title(out['choices'][0]['text'])
        if not new_title:
            return
        new_scope = generate_event_scope(llm_9b, scope_tmpl, new_title, " ".join(milestones[:8]))
        if new_scope:
            cursor.execute("UPDATE events SET title = ?, slug = ?, scope = ? WHERE id = ?",
                           (new_title, slugify(new_title), new_scope, event_id))
        else:
            cursor.execute("UPDATE events SET title = ?, slug = ? WHERE id = ?",
                           (new_title, slugify(new_title), event_id))
        logging.info(f"  [REFRAME] Event {event_id} re-titled from {len(milestones)} milestones: '{new_title}'")
    except Exception as e:
        logging.error(f"Reframe failed for event {event_id}: {e}")

def do_closure_write(cursor, encoder, event_id, evicted_article_ids, linked_event_ids):
    # Apply evictions
    for art_id in evicted_article_ids:
        evict_article_to_new_event(cursor, encoder, event_id, art_id)
        cursor.execute("UPDATE events SET article_count = article_count - 1 WHERE id = ?", (event_id,))
        
    # Apply saga linkages
    for other_id in linked_event_ids:
        assign_saga_id(cursor, event_id, other_id)
        
    # Close the event
    cursor.execute("UPDATE events SET state = 'closed' WHERE id = ?", (event_id,))

# Read prompt helper
def read_prompt(name):
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', f"{name}.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# Models loader
def load_models(dry_run=False):
    if dry_run:
        # Load MiniLM only
        from sentence_transformers import SentenceTransformer
        logging.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        return encoder, None, None

    from sentence_transformers import SentenceTransformer
    from llama_cpp import Llama

    logging.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    model_2b_path = os.environ.get('MODEL_2B_PATH', './models/gemma-2-2b-it-Q4_K_M.gguf')
    model_9b_path = os.environ.get('MODEL_GATE_PATH') or os.environ.get('MODEL_9B_PATH', './models/Qwen2.5-14B-Instruct-Q5_K_M.gguf')

    if not os.path.exists(model_2b_path):
        raise FileNotFoundError(f"Gemma 2B model not found at {model_2b_path}")
    if not os.path.exists(model_9b_path):
        raise FileNotFoundError(f"Gate model not found at {model_9b_path}")

    logging.info(f"Loading Gemma 2B from {model_2b_path}...")
    llm_2b = Llama(model_path=model_2b_path, n_ctx=2048, verbose=False)

    logging.info(f"Loading Qwen 14B from {model_9b_path}...")
    llm_9b = Llama(model_path=model_9b_path, n_ctx=2048, verbose=False)

    return encoder, llm_2b, llm_9b

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text)
    return text

# Small words that number-tokens attach to ("28-day", "Rs 5,000") — spelled-out
# numbers the model may legitimately rephrase are NOT policed; only digits.
def numbers_grounded(text, *sources):
    """True unless the text contains a digit-number that appears in NONE of the
    sources. Guards against small-model hallucinations like turning 'till
    July 20' into '28-day'. Compares on the bare digit string so '5,000' and
    '5000', '18' and '18-day' all match."""
    def digits(s):
        return set(re.findall(r'\d+', (s or '').replace(',', '')))
    src = set()
    for s in sources:
        src |= digits(s)
    for n in re.findall(r'\d+', (text or '').replace(',', '')):
        # Years and single/double digits phrased differently are common; only
        # flag numbers genuinely absent from every source.
        if n not in src:
            return False, n
    return True, None

def clean_title(raw):
    """Sanitize a generated event title. Qwen sometimes appends its reasoning
    in Chinese after the title ('...Oath Delay争议点在于...'), and spaceless CJK
    counts as ~1 'word', slipping past the word-count check. Keep only the
    first line, cut at the first non-Latin character, and require a sane
    result — otherwise return None (title regenerates on a later attach)."""
    if not raw:
        return None
    title = raw.strip().strip('"').strip("'").splitlines()[0]
    # Cut at the first character outside basic Latin + common punctuation
    m = re.search(r'[^\x20-\x7E‘’“”–—]', title)
    if m:
        title = title[:m.start()]
    title = title.strip(' -–—:;,.')
    words = title.split()
    if len(words) < 3 or len(words) > 12:
        return None
    return title

def run_write_burst_with_reconnect(conn, write_func, *args, **kwargs):
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            cursor = conn.cursor()
            res = write_func(cursor, *args, **kwargs)
            conn.commit()
            return conn, res
        except Exception as e:
            logging.error(f"Database write attempt {attempt+1} failed: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            if attempt < max_attempts - 1:
                logging.info("Reconnecting database and retrying write burst...")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_db_connection()
            else:
                raise e

def do_attach_write(cursor, matched_event_id, art_id, milestone, scraped_at, new_centroid, new_count, merged_keys, event_title, event_slug, linked_saga_event_ids):
    cursor.execute("""
        INSERT INTO event_articles (event_id, article_id, milestone, event_date)
        VALUES (?, ?, ?, ?)
    """, (matched_event_id, art_id, milestone, scraped_at))
    
    # COALESCE: attaches after the title/slug were first set pass None here —
    # they must never wipe the stored values (the old unconditional write
    # nulled the slug of every event on its 3rd+ attach, hiding it from the
    # frontend).
    # state='open': attaching an article REACTIVATES the event. A recently
    # closed story that gets fresh news reopens instead of spawning a twin.
    cursor.execute("""
        UPDATE events
        SET centroid = ?, last_seen = ?, article_count = ?, entity_keys = ?,
            title = COALESCE(?, title), slug = COALESCE(?, slug), state = 'open'
        WHERE id = ?
    """, (
        new_centroid.tobytes(),
        scraped_at,
        new_count,
        json.dumps(merged_keys),
        event_title,
        event_slug,
        matched_event_id
    ))
    
    for other_id in linked_saga_event_ids:
        assign_saga_id(cursor, matched_event_id, other_id)

    do_checkpoint_write(cursor, art_id, scraped_at)

def do_create_write(cursor, art_id, art_entities, embedding, scraped_at, scope):
    cursor.execute("""
        INSERT INTO events (title, slug, entity_keys, centroid, first_seen, last_seen, article_count, state, scope)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        None,
        None,
        json.dumps(art_entities),
        embedding.tobytes(),
        scraped_at,
        scraped_at,
        1,
        'open',
        scope
    ))
    new_ev_id = cursor.lastrowid
    
    cursor.execute("""
        INSERT INTO event_articles (event_id, article_id, milestone, event_date)
        VALUES (?, ?, ?, ?)
    """, (new_ev_id, art_id, None, scraped_at))
    
    do_checkpoint_write(cursor, art_id, scraped_at)
    return new_ev_id

def do_checkpoint_write(cursor, art_id, scraped_at=0):
    if SHARD_CTX:
        # Shards process in (scraped_at, id) order, so the cursor is the pair.
        cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id, last_scraped_at) VALUES (1, ?, ?)", (art_id, scraped_at))
    else:
        cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id) VALUES (1, ?)", (art_id,))

def run_final_seal(conn, encoder, llm_9b, closure_audit_prompt_template, saga_check_prompt_template,
                   title_prompt_template=None, event_scope_prompt_template=None):
    """Shard mode only: the shard's window is exhausted — run the full closure
    ceremony (exit audit + saga check) on EVERY remaining open event and close
    it. Deadline-aware and resumable: each closure commits individually, so an
    interrupted seal simply continues on the next run."""
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM events WHERE state = 'open' ORDER BY id ASC")
    open_ids = [r[0] for r in cursor.fetchall()]
    logging.info(f"[FINAL SEAL] {len(open_ids)} open events to close at end of shard window.")
    sealed_all = True
    for ev_id in open_ids:
        if DEADLINE_TS and time.time() >= DEADLINE_TS:
            logging.info("[FINAL SEAL] Deadline reached — sealing resumes on next run.")
            sealed_all = False
            break
        try:
            cursor = conn.cursor()
            evicted_ids, linked_ids = get_closure_decisions(
                cursor, llm_9b, ev_id,
                closure_audit_prompt_template, saga_check_prompt_template
            )
            conn, _ = run_write_burst_with_reconnect(
                conn,
                do_closure_write,
                encoder,
                ev_id,
                evicted_ids,
                linked_ids
            )
            if title_prompt_template and event_scope_prompt_template:
                def _reframe(cur):
                    reframe_event(cur, llm_9b, ev_id, title_prompt_template, event_scope_prompt_template)
                conn, _ = run_write_burst_with_reconnect(conn, _reframe)
            logging.info(f"[FINAL SEAL] Event {ev_id} closed.")
        except Exception as e:
            logging.error(f"[FINAL SEAL] Failed closing event {ev_id}: {e}")
            sealed_all = False
    return conn, sealed_all

def main():
    parser = argparse.ArgumentParser(description="Satya Timeline Service Pipeline")
    parser.add_argument('--dry-run', action='store_true', help="Run in dry-run mode without writing to DB or invoking LLMs")
    parser.add_argument('--from-id', type=int, help="Override database checkpoint and start from specific article ID")
    parser.add_argument('--limit', type=int, help="Limit number of articles to process in this run")
    parser.add_argument('--batch-size', type=int, default=25, help="Number of articles to process in this batch")
    parser.add_argument('--sim-threshold', type=float, default=0.60, help="Cosine similarity threshold for matches")
    parser.add_argument('--audit-event', type=int, help="Run closure audit calibration on a single event ID, printing KEEP/EVICT and exit without modifying database")
    parser.add_argument('--shard', type=int, help="Shard-replay mode: process only this shard's (scraped_at, id) window from the local snapshot, writing to a local shard DB. Zero Turso traffic.")
    parser.add_argument('--snapshot', type=str, default='./snapshot.db', help="Path to the local articles snapshot DB (shard mode)")
    parser.add_argument('--shards-config', type=str, default='./shards.json', help="Path to shards.json produced by export_snapshot.py (shard mode)")
    parser.add_argument('--deadline-ts', type=int, default=0, help="Unix timestamp; stop cleanly (between articles/closures) once passed. 0 = no deadline")
    parser.add_argument('--reset-events', action='store_true', help="DANGER: wipe events, event_articles and timeline_checkpoint, then replay every article from ID 0. Full coverage rebuild.")
    args = parser.parse_args()

    global DEADLINE_TS
    DEADLINE_TS = args.deadline_ts or 0

    if args.shard is not None:
        if args.from_id is not None:
            logging.critical("--from-id is not supported in shard mode.")
            sys.exit(1)
        if args.audit_event is not None:
            logging.critical("--audit-event is not supported in shard mode.")
            sys.exit(1)
        if not os.path.exists(args.shards_config):
            logging.critical(f"Shards config not found: {args.shards_config}")
            sys.exit(1)
        if not os.path.exists(args.snapshot):
            logging.critical(f"Snapshot DB not found: {args.snapshot}")
            sys.exit(1)
        with open(args.shards_config, 'r') as f:
            shards_cfg = json.load(f)
        matching = [s for s in shards_cfg.get('shards', []) if s.get('shard') == args.shard]
        if not matching:
            logging.critical(f"Shard {args.shard} not found in {args.shards_config}")
            sys.exit(1)
        w = matching[0]
        SHARD_CTX.update({
            'shard': args.shard,
            'shard_db': f"./shard_{args.shard}.db",
            'snapshot_db': args.snapshot,
            'from_ts': int(w['from_ts']), 'from_id': int(w['from_id']),
            'to_ts': int(w['to_ts']), 'to_id': int(w['to_id']),
        })
        logging.info(f"[SHARD {args.shard}] Window: ({w['from_ts']}, id {w['from_id']}) .. ({w['to_ts']}, id {w['to_id']}) exclusive | count={w.get('count')}")

    # Environment variables override BATCH_SIZE
    env_batch_size = os.environ.get('BATCH_SIZE')
    batch_size = int(env_batch_size) if env_batch_size else args.batch_size
    if args.limit:
        batch_size = min(batch_size, args.limit)

    logging.info(f"Pipeline started. dry_run={args.dry_run}, batch_size={batch_size}, sim_threshold={args.sim_threshold}")

    # Connect to DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        sys.exit(1)

    # Full-coverage rebuild: wipe timeline state so replay starts at article 0.
    if args.reset_events:
        if SHARD_CTX:
            logging.critical("--reset-events is not supported in shard mode.")
            conn.close()
            sys.exit(1)
        if args.dry_run:
            logging.critical("--reset-events with --dry-run makes no sense. Aborting.")
            conn.close()
            sys.exit(1)
        logging.warning("RESET: wiping events, event_articles and timeline_checkpoint...")
        try:
            cursor.execute("DELETE FROM event_articles")
            cursor.execute("DELETE FROM events")
            cursor.execute("DELETE FROM timeline_checkpoint")
            conn.commit()
            logging.warning("RESET complete. Replay starts from article ID 0.")
        except Exception as e:
            logging.critical(f"Reset failed: {e}")
            conn.close()
            sys.exit(1)

    # Shard already sealed? (idempotent re-dispatch safety)
    if SHARD_CTX:
        try:
            cursor.execute("SELECT shard_done FROM shard_meta WHERE id = 1")
            row = cursor.fetchone()
            if row and int(row[0] or 0) == 1:
                logging.info(f"[SHARD {SHARD_CTX['shard']}] Already sealed — nothing to do.")
                print("has_more=false")
                print("articles_attached=0")
                print("shard_done=true")
                conn.close()
                sys.exit(0)
        except Exception as e:
            logging.error(f"Failed to read shard_meta: {e}")

    # 1. Cursor Loading / Audit Event check
    if args.audit_event is not None:
        try:
            # Load models (only 9b is needed)
            _, _, llm_9b = load_models(dry_run=False)
            closure_audit_prompt_template = read_prompt('closure_audit')
        except Exception as e:
            logging.critical(f"Failed to load models/prompts for calibration: {e}")
            conn.close()
            sys.exit(1)
            
        logging.info(f"Running audit calibration for Event ID {args.audit_event}...")
        # Fetch event info
        cursor.execute("SELECT title FROM events WHERE id = ?", (args.audit_event,))
        row = cursor.fetchone()
        if not row:
            logging.critical(f"Event ID {args.audit_event} not found.")
            conn.close()
            sys.exit(1)
        event_title = row[0] or "Untitled Event"
        founding_milestone = get_founding_milestone(cursor, args.audit_event)
        
        # Fetch member articles
        cursor.execute("""
            SELECT ea.article_id, ea.milestone, a.title, a.rephrased_title
            FROM event_articles ea
            JOIN articles a ON ea.article_id = a.id
            WHERE ea.event_id = ?
        """, (args.audit_event,))
        member_articles = cursor.fetchall()
        
        print(f"Auditing Event {args.audit_event} - '{event_title}'")
        print(f"Founding Milestone: '{founding_milestone}'")
        print(f"Total articles to audit: {len(member_articles)}")
        
        keep_count = 0
        evict_count = 0
        for art_id, milestone, orig_title, reph_title in member_articles:
            art_milestone = milestone or reph_title or orig_title or "No Title"
            art_title = reph_title or orig_title or "No Title"
            
            prompt = closure_audit_prompt_template.format(
                event_title=event_title,
                founding_milestone=founding_milestone,
                article_title=art_title,
                milestone=art_milestone
            )
            output = llm_9b(prompt, max_tokens=100, stop=["<|im_end|>"], temperature=0.0)
            response_text = output['choices'][0]['text'].strip()
            decision = "KEEP" if not response_text.upper().startswith("EVICT") else "EVICT"
            
            if decision == "KEEP":
                keep_count += 1
            else:
                evict_count += 1
                
            print(f"Article ID {art_id:5d}: {decision:5s} | Title: {art_title[:80]} | Milestone: {art_milestone[:80]} | LLM Raw: {response_text.strip()}")
            
        print("\n--- AUDIT SUMMARY ---")
        print(f"Total Articles: {len(member_articles)}")
        print(f"KEEP: {keep_count}")
        print(f"EVICT: {evict_count}")
        print("has_more=false")
        print("articles_attached=0")
        conn.close()
        sys.exit(0)

    start_id = 0
    start_sa = 0
    if args.from_id is not None:
        start_id = args.from_id
        logging.info(f"Overriding cursor. Starting from article ID: {start_id}")
    else:
        try:
            if SHARD_CTX:
                cursor.execute("SELECT last_article_id, last_scraped_at FROM timeline_checkpoint WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    start_id = int(row[0])
                    start_sa = int(row[1] or 0)
                    logging.info(f"[SHARD] Loaded cursor: (scraped_at {start_sa}, id {start_id})")
                else:
                    start_id = -1
                    start_sa = 0
                    logging.info("[SHARD] Fresh shard — no checkpoint yet.")
            else:
                cursor.execute("SELECT last_article_id FROM timeline_checkpoint WHERE id = 1")
                row = cursor.fetchone()
                if row:
                    start_id = int(row[0])
                    logging.info(f"Loaded cursor from DB. Last processed article ID: {start_id}")
                else:
                    logging.info("Checkpoint empty. Starting from article ID 0.")
        except Exception as e:
            logging.error(f"Failed to read checkpoint table: {e}. Starting from ID 0.")

    # 2. Fetch Batch
    logging.info(f"Fetching up to {batch_size} articles after cursor (id {start_id})...")
    try:
        if SHARD_CTX:
            # Shard mode: snapshot is pre-filtered for eligibility; process in
            # strict (scraped_at, id) order within the shard's window.
            # Condition 1 = cursor (exclusive), 2 = window from (inclusive),
            # 3 = window to (exclusive).
            cursor.execute("""
                SELECT a.id, a.title, a.rephrased_title, a.rephrased_article, a.scraped_at,
                       a.party_mentioned, a.ministers_mentioned, a.states_mentioned, a.cities_mentioned, a.civic_flag
                FROM articles a
                WHERE (a.scraped_at > ? OR (a.scraped_at = ? AND a.id > ?))
                  AND (a.scraped_at > ? OR (a.scraped_at = ? AND a.id >= ?))
                  AND (a.scraped_at < ? OR (a.scraped_at = ? AND a.id < ?))
                ORDER BY a.scraped_at ASC, a.id ASC
                LIMIT ?
            """, (start_sa, start_sa, start_id,
                  SHARD_CTX['from_ts'], SHARD_CTX['from_ts'], SHARD_CTX['from_id'],
                  SHARD_CTX['to_ts'], SHARD_CTX['to_ts'], SHARD_CTX['to_id'],
                  batch_size))
        else:
            # NOT-IN clause: idempotency. Articles already filed (e.g. by the
            # Timeline Doctor ahead of this cursor) are stepped over instead of
            # being processed twice. Tracked-figure title matches are eligible
            # even without entity tags.
            tf_clause, tf_params = tracked_match_sql()
            cursor.execute(f"""
                SELECT a.id, a.title, a.rephrased_title, a.rephrased_article, a.scraped_at,
                       a.party_mentioned, a.ministers_mentioned, a.states_mentioned, a.cities_mentioned, a.civic_flag
                FROM articles a
                WHERE a.id > ?
                  AND (({ELIGIBILITY_SQL}) OR {tf_clause})
                  AND a.id NOT IN (SELECT article_id FROM event_articles)
                ORDER BY a.id ASC
                LIMIT ?
            """, (start_id, *tf_params, batch_size))
        articles_rows = cursor.fetchall()
    except Exception as e:
        logging.critical(f"Failed to fetch articles batch: {e}")
        conn.close()
        sys.exit(1)

    if not articles_rows:
        logging.info("No articles to process in this batch.")
        if not SHARD_CTX:
            print("has_more=false")
            print("articles_attached=0")
            conn.close()
            sys.exit(0)
        # Shard mode: window exhausted, but the final seal may still be
        # pending — fall through (models load, matching loop no-ops, and the
        # end-of-window seal below finishes the job).

    # 3. Load LLM and Embedding models
    try:
        encoder, llm_2b, llm_9b = load_models(args.dry_run)
    except Exception as e:
        logging.critical(f"Failed to load models: {e}")
        conn.close()
        sys.exit(1)

    # Read prompt templates if not in dry-run mode
    attach_prompt_template = ""
    event_scope_prompt_template = ""
    milestone_prompt_template = ""
    title_prompt_template = ""
    closure_audit_prompt_template = ""
    saga_check_prompt_template = ""
    if not args.dry_run:
        try:
            attach_prompt_template = read_prompt('attach_gate')
            event_scope_prompt_template = read_prompt('event_scope')
            milestone_prompt_template = read_prompt('milestone')
            verify_milestone_prompt_template = read_prompt('verify_milestone')
            title_prompt_template = read_prompt('event_title')
            closure_audit_prompt_template = read_prompt('closure_audit_batch')
            saga_check_prompt_template = read_prompt('saga_check')
        except Exception as e:
            logging.critical(f"Failed to load prompt templates: {e}")
            conn.close()
            sys.exit(1)

    # 4a. HOUSEKEEPING FIRST — close overdue events BEFORE filing new articles.
    # Closures are a finite queue; the article stream is infinite. Running
    # filing first starves closures forever during a backlog catch-up.
    # Logical clock = scraped_at of the bookmark article (last one processed),
    # since this run's batch hasn't been processed yet.
    if not args.dry_run:
        clock_ts = 0
        if SHARD_CTX:
            clock_ts = start_sa
        elif start_id > 0:
            try:
                cursor.execute("SELECT scraped_at FROM articles WHERE id = ?", (start_id,))
                row = cursor.fetchone()
                clock_ts = int(row[0]) if row and row[0] is not None else 0
            except Exception as e:
                logging.error(f"Failed to read bookmark article's scraped_at for logical clock: {e}")

        if clock_ts > 0:
            closure_cutoff = clock_ts - 21 * 24 * 3600
            logging.info(f"\nRunning chapter closure sweep relative to logical clock cutoff: {closure_cutoff}...")

            # One-off migration of dormant events to closed state
            try:
                cursor = conn.cursor()
                cursor.execute("UPDATE events SET state = 'closed' WHERE state = 'dormant'")
                migrated = cursor.rowcount
                if migrated > 0:
                    logging.info(f"Migrated {migrated} dormant events to closed state.")
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to migrate dormant events: {e}")
                try:
                    conn.rollback()
                except:
                    pass

            # Close events INACTIVE for 21+ days (last_seen), not events merely
            # founded 21+ days ago. A story with recent news stays open and
            # keeps growing; only genuinely dormant stories conclude.
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM events WHERE state = 'open' AND last_seen < ?", (closure_cutoff,))
                events_to_close = [r[0] for r in cursor.fetchall()]

                if events_to_close:
                    logging.info(f"Found {len(events_to_close)} events to close.")
                    for ev_id in events_to_close:
                        if DEADLINE_TS and time.time() >= DEADLINE_TS:
                            logging.info("Deadline reached — remaining closures resume on next run.")
                            break
                        logging.info(f"Closing Event ID {ev_id}...")
                        try:
                            # Refresh cursor for reads
                            cursor = conn.cursor()
                            # Phase 1: All LLM decisions gathered in memory (no DB writes)
                            evicted_ids, linked_ids = get_closure_decisions(
                                cursor, llm_9b, ev_id,
                                closure_audit_prompt_template, saga_check_prompt_template
                            )

                            # Phase 2: Apply all changes in one write transaction burst
                            conn, _ = run_write_burst_with_reconnect(
                                conn,
                                do_closure_write,
                                encoder,
                                ev_id,
                                evicted_ids,
                                linked_ids
                            )
                            # Re-derive title/scope from the whole story (fixes framing drift)
                            def _reframe(cur):
                                reframe_event(cur, llm_9b, ev_id, title_prompt_template, event_scope_prompt_template)
                            conn, _ = run_write_burst_with_reconnect(conn, _reframe)
                            cursor = conn.cursor()
                            logging.info(f"Event ID {ev_id} closed successfully.")
                        except Exception as e:
                            logging.error(f"Failed to run closure ceremony for Event ID {ev_id}: {e}")
            except Exception as e:
                logging.error(f"Failed to query events for closure sweep: {e}")

    # 4. Load candidate events into memory: OPEN events plus recently-active
    # CLOSED events (last activity within the reopen window). A developing
    # story whose event was auto-closed must be able to REOPEN and keep
    # growing, instead of spawning a duplicate — otherwise long timelines
    # fragment every time the 21-day clock closes a still-live story.
    REOPEN_WINDOW = 21 * 24 * 3600
    batch_max_sa = max((int(r[4]) for r in articles_rows if r[4] is not None), default=0)
    closed_cutoff = batch_max_sa - REOPEN_WINDOW if batch_max_sa else 0
    open_events = []
    try:
        cursor.execute("""
            SELECT id, title, entity_keys, centroid, last_seen, article_count, first_seen, scope, state
            FROM events
            WHERE state = 'open' OR (state = 'closed' AND last_seen >= ?)
        """, (closed_cutoff,))
        event_rows = cursor.fetchall()
        reopenable = 0
        for r in event_rows:
            try:
                open_events.append({
                    'id': int(r[0]),
                    'title': r[1],
                    'entity_keys': set(json.loads(r[2])),
                    'centroid': np.frombuffer(r[3], dtype=np.float32),
                    'last_seen': int(r[4]) if r[4] is not None else 0,
                    'article_count': int(r[5]),
                    'first_seen': int(r[6]) if r[6] is not None else 0,
                    'scope': r[7],
                    'state': r[8],
                })
                if r[8] == 'closed':
                    reopenable += 1
            except Exception as ex:
                logging.error(f"Error parsing event row {r[0]}: {ex}")
        logging.info(f"Loaded {len(open_events)} candidate events ({reopenable} recently-closed, reopenable).")
    except Exception as e:
        logging.critical(f"Failed to load candidate events: {e}")
        conn.close()
        sys.exit(1)

    attached_count = 0
    last_processed_id = start_id
    last_processed_sa = start_sa
    max_scraped_at = 0

    # 5. Pipeline Matching Loop
    for row in articles_rows:
        if DEADLINE_TS and time.time() >= DEADLINE_TS:
            logging.info("Deadline reached — stopping batch early (per-article checkpoints already committed).")
            break
        art_id = int(row[0])
        orig_title = row[1] or ""
        reph_title = row[2] or orig_title
        decompressed_article = decompress_text(row[3])
        scraped_at = int(row[4]) if row[4] is not None else 0
        max_scraped_at = max(max_scraped_at, scraped_at)
        last_processed_id = art_id
        last_processed_sa = scraped_at

        # Parse entities (+ stamp tracked-figure keys so matching works even
        # when the classifier didn't tag the person)
        art_entities = list(extract_keys(row[5], row[6], row[7], row[8]))
        for tk in tracked_keys_for(f"{reph_title} {orig_title}"):
            if tk not in art_entities:
                art_entities.append(tk)

        logging.info(f"\nProcessing article ID {art_id}: '{reph_title}'")
        logging.info(f"  Entities: {art_entities}")

        try:
            # Step 1: Embed title + summary
            combined_text = f"{reph_title} {decompressed_article}".strip()
            embedding = encoder.encode(combined_text, convert_to_numpy=True)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            best_match = None
            best_sim = -1.0

            # Step 2: Filter candidates in memory. Candidacy keys off LAST
            # ACTIVITY, not founding date — an event stays eligible as long as
            # it received news within the last 21 days, so a continuously
            # covered story grows into one long timeline instead of chaptering.
            cutoff_time = scraped_at - 21 * 24 * 3600
            for ev in open_events:
                if ev['last_seen'] > cutoff_time:
                    shared_keys = ev['entity_keys'].intersection(art_entities)
                    if shared_keys:
                        sim = float(np.dot(ev['centroid'], embedding))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = ev

            logging.info(f"  Best similarity match: {best_sim:.4f} with Event ID {best_match['id'] if best_match else None}")

            # Do all database read operations BEFORE starting any LLM calls
            existing_milestones = []
            if best_sim >= args.sim_threshold and best_match is not None:
                if not args.dry_run:
                    try:
                        # If count is 1, adding the new article will make it 2, triggering title generation
                        if best_match['article_count'] == 1:
                            cursor.execute("""
                                SELECT milestone FROM event_articles 
                                WHERE event_id = ? 
                                ORDER BY event_date ASC
                            """, (best_match['id'],))
                            existing_milestones = [r[0] for r in cursor.fetchall() if r[0]]
                    except Exception as e:
                        logging.error(f"Failed to pre-fetch milestones for candidate: {e}")

            matched_event = None
            if best_sim >= args.sim_threshold and best_match is not None:
                if scraped_at < best_match['first_seen']:
                    logging.info(f"  [REJECT] Date-guard: article predates event {best_match['id']} (scraped_at {scraped_at} < first_seen {best_match['first_seen']}). Skip gate call.")
                elif args.dry_run:
                    logging.info(f"  [DRY-RUN] Simulating Gemma 9B attach gate approval for Event ID {best_match['id']} (sim: {best_sim:.4f})")
                    matched_event = best_match
                else:
                    event_name = best_match['title'] or "Untitled Event"
                    founding_milestone = get_founding_milestone(cursor, best_match['id'])
                    
                    # Fetch recent milestones (last 2)
                    cursor.execute("""
                        SELECT milestone FROM event_articles 
                        WHERE event_id = ? AND milestone IS NOT NULL AND milestone != ''
                        ORDER BY event_date DESC LIMIT 2
                    """, (best_match['id'],))
                    milestones_list = [row[0] for row in cursor.fetchall()]
                    milestones_list.reverse()
                    if not milestones_list:
                        milestones_list = [founding_milestone]
                    recent_milestones_str = "\n".join(f"- {m}" for m in milestones_list)
                    
                    scope_val = best_match.get('scope') or founding_milestone

                    event_began_str = datetime.fromtimestamp(best_match['first_seen']).strftime("%Y-%m-%d") if best_match['first_seen'] else "N/A"
                    article_date_str = datetime.fromtimestamp(scraped_at).strftime("%Y-%m-%d") if scraped_at else "N/A"

                    # Format Single attach gate prompt
                    prompt = attach_prompt_template.format(
                        event_title=event_name,
                        scope=scope_val,
                        event_began=event_began_str,
                        recent_milestones=recent_milestones_str,
                        article_date=article_date_str,
                        new_title=reph_title,
                        new_summary=decompressed_article[:800]
                    )

                    # Invoke gate model for Vote (reasoning-first prompt: the
                    # verdict is the LAST ATTACH/REJECT token, not the first
                    # word). One bad article must not crash the whole batch —
                    # fail-safe to REJECT (new event) and keep going.
                    try:
                        output = llm_9b(prompt, max_tokens=350, stop=["<|im_end|>"], temperature=0.0)
                        response_text = output['choices'][0]['text'].strip()
                        verdict_matches = re.findall(r'\b(ATTACH|REJECT)\b', response_text.upper())
                        vote = verdict_matches[-1] if verdict_matches else "REJECT"
                    except Exception as gate_err:
                        response_text = ""
                        vote = "REJECT"
                        logging.error(f"Gate call failed for article_id={art_id}, event_id={best_match['id']}: {type(gate_err).__name__}: {gate_err}. Defaulting to REJECT.")

                    logging.info(f"Nomination: article_id={art_id}, event_id={best_match['id']}, cosine={best_sim:.4f}, vote={vote} | Raw: {response_text}")

                    # Validation
                    if vote == "ATTACH":
                        matched_event = best_match
                        logging.info(f"  [ATTACH] Approved by scope gate.")
                    else:
                        logging.info(f"  [REJECT] Rejected by scope gate.")

            # Step 3: Attach or Create Event
            if matched_event is not None:
                # ATTACH
                if args.dry_run:
                    # Simulate centroid EMA update in memory
                    new_centroid = 0.7 * matched_event['centroid'] + 0.3 * embedding
                    new_centroid_norm = np.linalg.norm(new_centroid)
                    if new_centroid_norm > 0:
                        new_centroid /= new_centroid_norm
                    
                    matched_event['centroid'] = new_centroid
                    matched_event['last_seen'] = scraped_at
                    matched_event['article_count'] += 1
                    # Note: first_seen doesn't change on attach
                    
                    # Keep existing keys that also appear in new article first, cap at 15 keys
                    existing_set = matched_event['entity_keys']
                    new_set = set(art_entities)
                    intersection_list = list(existing_set.intersection(new_set))
                    remaining_new = list(new_set.difference(existing_set))
                    remaining_existing = list(existing_set.difference(new_set))
                    merged_keys_list = (intersection_list + remaining_new + remaining_existing)[:15]
                    matched_event['entity_keys'] = set(merged_keys_list)
                    
                    attached_count += 1
                    logging.info(f"  [DRY-RUN] Simulated attach to Event ID {matched_event['id']}. New count: {matched_event['article_count']}")
                else:
                    milestone = None
                    # Write Milestone using Gemma 2B with Actor-Critic Verification Loop
                    MAX_RETRIES = 3
                    for attempt in range(MAX_RETRIES):
                        prompt = milestone_prompt_template.format(
                            title=orig_title,
                            summary=decompressed_article[:1200]
                        )
                        output = llm_2b(prompt, max_tokens=100, stop=["<end_of_turn>"], temperature=0.1 + (0.15 * attempt))
                        gen_milestone = output['choices'][0]['text'].strip()

                        # Validate milestone (max 30 words, no quotes, numbers grounded)
                        words = gen_milestone.split()
                        valid = True
                        if len(words) < 3 or len(words) > 30:
                            valid = False
                        if gen_milestone.startswith('"') or gen_milestone.endswith('"') or gen_milestone.startswith("'") or gen_milestone.endswith("'"):
                            valid = False
                        grounded, bad_num = numbers_grounded(gen_milestone, orig_title, decompressed_article)
                        if not grounded:
                            valid = False
                            logging.info(f"  Milestone rejected: hallucinated number '{bad_num}' not in source.")

                        # Verification using 2B Model (Critic)
                        if valid:
                            verify_prompt = verify_milestone_prompt_template.format(
                                summary=decompressed_article[:1200],
                                milestone=gen_milestone
                            )
                            verify_out = llm_2b(verify_prompt, max_tokens=10, stop=["<end_of_turn>"], temperature=0.0)
                            verdict = verify_out['choices'][0]['text'].strip().upper()
                            if "FAIL" in verdict:
                                valid = False
                                logging.info(f"  Milestone validation failed (Critic rejected Attempt {attempt+1}/{MAX_RETRIES}): '{gen_milestone}'")

                        if valid:
                            milestone = gen_milestone
                            logging.info(f"  Generated Milestone (Attempt {attempt+1}): '{milestone}'")
                            break
                        else:
                            if attempt == MAX_RETRIES - 1:
                                logging.info(f"  Milestone validation failed after {MAX_RETRIES} attempts. Falling back to 14B model.")
                                fallback_prompt = milestone_prompt_template.format(
                                    title=orig_title,
                                    summary=decompressed_article[:1200]
                                )
                                fb_out = llm_9b(fallback_prompt, max_tokens=100, stop=["<end_of_turn>", "<|im_end|>"], temperature=0.1)
                                fallback_milestone = fb_out['choices'][0]['text'].strip()
                                if fallback_milestone.startswith('"') and fallback_milestone.endswith('"'):
                                    fallback_milestone = fallback_milestone[1:-1]
                                milestone = fallback_milestone
                                logging.info(f"  Generated Milestone (14B Fallback): '{milestone}'")

                    # Calculate EMA centroid and new count
                    new_centroid = 0.7 * matched_event['centroid'] + 0.3 * embedding
                    new_centroid_norm = np.linalg.norm(new_centroid)
                    if new_centroid_norm > 0:
                        new_centroid = new_centroid / new_centroid_norm

                    new_count = matched_event['article_count'] + 1
                    # Keep existing keys that also appear in new article first, cap at 15 keys
                    existing_set = matched_event['entity_keys']
                    new_set = set(art_entities)
                    intersection_list = list(existing_set.intersection(new_set))
                    remaining_new = list(new_set.difference(existing_set))
                    remaining_existing = list(existing_set.difference(new_set))
                    merged_keys = (intersection_list + remaining_new + remaining_existing)[:15]

                    # If count hits 2, generate Title using Gemma 9B
                    event_title = matched_event['title']
                    event_slug = None
                    if new_count >= 2 and not matched_event['title']:
                        all_ms = list(existing_milestones)
                        if milestone:
                            all_ms.append(milestone)
                        if not all_ms:
                            all_ms = [reph_title]
                        
                        title_prompt = title_prompt_template.format(milestones="\n".join(f"- {m}" for m in all_ms))
                        title_out = llm_9b(title_prompt, max_tokens=60, stop=["<|im_end|>"], temperature=0.0)
                        gen_title = clean_title(title_out['choices'][0]['text'])

                        if gen_title:
                            event_title = gen_title
                            event_slug = slugify(event_title)
                            logging.info(f"  Generated Event Title: '{event_title}' | slug: {event_slug}")
                        else:
                            logging.error(f"  Generated Title failed sanitation: '{title_out['choices'][0]['text'][:120]}'. Keeping NULL.")

                    # Run Saga check at visibility (when count transitions to 2 articles and it gets a title)
                    linked_saga_event_ids = []
                    if new_count == 2 and event_title:
                        try:
                            linked_saga_event_ids = check_saga_links_in_memory(
                                cursor, llm_9b, matched_event['id'], event_title,
                                merged_keys, matched_event['first_seen'], scraped_at,
                                saga_check_prompt_template
                            )
                        except Exception as e:
                            logging.error(f"Saga link check in memory failed: {e}")

                    # Write to database (transactional execution per article using hot reconnect wrapper)
                    conn, _ = run_write_burst_with_reconnect(
                        conn,
                        do_attach_write,
                        matched_event['id'],
                        art_id,
                        milestone,
                        scraped_at,
                        new_centroid,
                        new_count,
                        merged_keys,
                        event_title,
                        event_slug,
                        linked_saga_event_ids
                    )
                    # Refresh cursor after reconnect/retry block in case it reconnected
                    cursor = conn.cursor()

                    # Update memory cache
                    matched_event['centroid'] = new_centroid
                    matched_event['last_seen'] = scraped_at
                    matched_event['article_count'] = new_count
                    matched_event['entity_keys'] = set(merged_keys)
                    matched_event['title'] = event_title
                    
                    attached_count += 1
                    logging.info(f"  Attached to Event ID {matched_event['id']} successfully.")
            else:
                # CREATE NEW EVENT
                if args.dry_run:
                    new_ev_id = 100000 + len(open_events)
                    open_events.append({
                        'id': new_ev_id,
                        'title': f"Simulated Event {new_ev_id}",
                        'entity_keys': set(art_entities),
                        'centroid': embedding,
                        'last_seen': scraped_at,
                        'article_count': 1,
                        'first_seen': scraped_at,
                        'scope': None
                    })
                    attached_count += 1
                    logging.info(f"  [DRY-RUN] Simulated create new Event ID {new_ev_id}.")
                else:
                    # Generate scope charter using Gemma 9B
                    scope = generate_event_scope(
                        llm_9b, event_scope_prompt_template,
                        reph_title, decompressed_article
                    )
                    if scope:
                        logging.info(f"  Generated Event Scope: '{scope}'")
                    else:
                        logging.warning("  Failed to generate event scope. Storing NULL.")

                    # Write to database (transactional execution per article using hot reconnect wrapper)
                    conn, new_ev_id = run_write_burst_with_reconnect(
                        conn,
                        do_create_write,
                        art_id,
                        art_entities,
                        embedding,
                        scraped_at,
                        scope
                    )
                    cursor = conn.cursor()

                    # Update memory cache
                    open_events.append({
                        'id': new_ev_id,
                        'title': None,
                        'entity_keys': set(art_entities),
                        'centroid': embedding,
                        'last_seen': scraped_at,
                        'article_count': 1,
                        'first_seen': scraped_at,
                        'scope': scope
                    })
                    attached_count += 1
                    logging.info(f"  Created new invisible Event ID {new_ev_id}.")

        except Exception as err:
            # Poison pill guard: skip this article, advance cursor, and continue
            logging.error(f"Poison-pill encountered for article ID {art_id}: {err}")
            if not args.dry_run:
                try:
                    conn, _ = run_write_burst_with_reconnect(
                        conn,
                        do_checkpoint_write,
                        art_id,
                        scraped_at
                    )
                    cursor = conn.cursor()
                except Exception as commit_err:
                    logging.critical(f"Failed to advance cursor after poison-pill: {commit_err}")
                    conn.close()
                    sys.exit(1)
            continue

    # (Chapter closure sweep now runs BEFORE the matching loop — see
    # "HOUSEKEEPING FIRST" above. Finite work must outrank the infinite stream.)

    # Check if there are more articles remaining in the table (with reconnect logic)
    has_more = "false"
    for attempt in range(2):
        try:
            cursor = conn.cursor()
            if SHARD_CTX:
                cursor.execute("""
                    SELECT a.id FROM articles a
                    WHERE (a.scraped_at > ? OR (a.scraped_at = ? AND a.id > ?))
                      AND (a.scraped_at > ? OR (a.scraped_at = ? AND a.id >= ?))
                      AND (a.scraped_at < ? OR (a.scraped_at = ? AND a.id < ?))
                    LIMIT 1
                """, (last_processed_sa, last_processed_sa, last_processed_id,
                      SHARD_CTX['from_ts'], SHARD_CTX['from_ts'], SHARD_CTX['from_id'],
                      SHARD_CTX['to_ts'], SHARD_CTX['to_ts'], SHARD_CTX['to_id']))
            else:
                tf_clause, tf_params = tracked_match_sql()
                cursor.execute(f"""
                    SELECT a.id FROM articles a
                    WHERE a.id > ?
                      AND (({ELIGIBILITY_SQL}) OR {tf_clause})
                      AND a.id NOT IN (SELECT article_id FROM event_articles)
                    LIMIT 1
                """, (last_processed_id, *tf_params))
            row = cursor.fetchone()
            if row:
                has_more = "true"
            break
        except Exception as e:
            logging.error(f"Failed to check for remaining articles (attempt {attempt+1}): {e}")
            if attempt == 0:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_db_connection()

    # Shard mode: end-of-window final seal — when the window is exhausted,
    # close ALL remaining open events (full ceremony), then mark the shard done.
    shard_done = "false"
    if SHARD_CTX and not args.dry_run and has_more == "false":
        conn, sealed_all = run_final_seal(
            conn, encoder, llm_9b,
            closure_audit_prompt_template, saga_check_prompt_template,
            title_prompt_template, event_scope_prompt_template
        )
        if sealed_all:
            def _mark_shard_done(cur):
                cur.execute("INSERT OR REPLACE INTO shard_meta (id, shard_done) VALUES (1, 1)")
            conn, _ = run_write_burst_with_reconnect(conn, _mark_shard_done)
            shard_done = "true"
            logging.info(f"[SHARD {SHARD_CTX['shard']}] SEALED — shard complete.")

    # Close connection
    try:
        conn.close()
    except Exception:
        pass

    logging.info(f"Batch completed. attached={attached_count}, has_more={has_more}")
    
    # Dry Run summary printing
    if args.dry_run:
        logging.info("\n=== DRY-RUN GROUPING SUMMARY ===")
        sorted_events = sorted(open_events, key=lambda e: e['article_count'], reverse=True)
        logging.info(f"Total Simulated Events: {len(open_events)}")
        logging.info("Top 20 Events by Simulated Article Count:")
        for ev in sorted_events[:20]:
            logging.info(f"  Event ID {ev['id']}: Count: {ev['article_count']} | Last Seen: {ev['last_seen']} | Entities: {list(ev['entity_keys'])}")

    # Print output to stdout for GHA step capture
    print(f"has_more={has_more}")
    print(f"articles_attached={attached_count}")
    if SHARD_CTX:
        print(f"shard_done={shard_done}")

if __name__ == '__main__':
    main()
