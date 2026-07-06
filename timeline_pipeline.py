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

def get_db_connection():
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
        output = llm_9b(prompt, max_tokens=150, stop=["<end_of_turn>"], temperature=0.0)
        scope = output['choices'][0]['text'].strip()
        
        if not scope:
            return None
        words = scope.split()
        if len(words) > 80:
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
    
    for c_id, c_title, c_keys_json, c_first, c_last, c_saga, c_scope in closed_events:
        if c_id == event_id:
            continue
        if not c_title:
            continue
        # Skip if they already share the same saga_id
        if t_saga is not None and c_saga is not None and t_saga == c_saga:
            continue
            
        c_keys = set(json.loads(c_keys_json))
        if len(t_keys.intersection(c_keys)) >= 2:
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
            
            output = llm_9b(prompt, max_tokens=100, stop=["<end_of_turn>"], temperature=0.0)
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
        
        for art_id, milestone, orig_title, reph_title in member_articles:
            art_milestone = milestone or reph_title or orig_title or "No Title"
            art_title = reph_title or orig_title or "No Title"
            
            prompt = closure_audit_prompt_template.format(
                event_title=event_title,
                scope=scope_val,
                article_title=art_title,
                milestone=art_milestone
            )
            
            output = llm_9b(prompt, max_tokens=100, stop=["<end_of_turn>"], temperature=0.0)
            res_text = output['choices'][0]['text'].strip().upper()
            
            if res_text.startswith("EVICT"):
                logging.info(f"[EVICT DECISION] Event {event_id}: Evict Article {art_id}")
                evicted_article_ids.append(art_id)
            else:
                logging.info(f"[KEEP DECISION] Event {event_id}: Keep Article {art_id}")
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
    model_9b_path = os.environ.get('MODEL_9B_PATH', './models/gemma-2-9b-it-Q6_K.gguf')

    if not os.path.exists(model_2b_path):
        raise FileNotFoundError(f"Gemma 2B model not found at {model_2b_path}")
    if not os.path.exists(model_9b_path):
        raise FileNotFoundError(f"Gemma 9B model not found at {model_9b_path}")

    logging.info(f"Loading Gemma 2B from {model_2b_path}...")
    llm_2b = Llama(model_path=model_2b_path, n_ctx=2048, verbose=False)

    logging.info(f"Loading Gemma 9B from {model_9b_path}...")
    llm_9b = Llama(model_path=model_9b_path, n_ctx=2048, verbose=False)

    return encoder, llm_2b, llm_9b

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '-', text)
    return text

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
    
    cursor.execute("""
        UPDATE events 
        SET centroid = ?, last_seen = ?, article_count = ?, entity_keys = ?, title = ?, slug = ?
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
        
    cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id) VALUES (1, ?)", (art_id,))

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
    
    cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id) VALUES (1, ?)", (art_id,))
    return new_ev_id

def do_checkpoint_write(cursor, art_id):
    cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id) VALUES (1, ?)", (art_id,))

def main():
    parser = argparse.ArgumentParser(description="Satya Timeline Service Pipeline")
    parser.add_argument('--dry-run', action='store_true', help="Run in dry-run mode without writing to DB or invoking LLMs")
    parser.add_argument('--from-id', type=int, help="Override database checkpoint and start from specific article ID")
    parser.add_argument('--limit', type=int, help="Limit number of articles to process in this run")
    parser.add_argument('--batch-size', type=int, default=25, help="Number of articles to process in this batch")
    parser.add_argument('--sim-threshold', type=float, default=0.60, help="Cosine similarity threshold for matches")
    parser.add_argument('--audit-event', type=int, help="Run closure audit calibration on a single event ID, printing KEEP/EVICT and exit without modifying database")
    args = parser.parse_args()

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
            output = llm_9b(prompt, max_tokens=100, stop=["<end_of_turn>"], temperature=0.0)
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
    if args.from_id is not None:
        start_id = args.from_id
        logging.info(f"Overriding cursor. Starting from article ID: {start_id}")
    else:
        try:
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
    logging.info(f"Fetching up to {batch_size} articles after ID {start_id}...")
    try:
        cursor.execute("""
            SELECT a.id, a.title, a.rephrased_title, a.rephrased_article, a.scraped_at,
                   a.party_mentioned, a.ministers_mentioned, a.states_mentioned, a.cities_mentioned, a.civic_flag
            FROM articles a
            WHERE a.id > ?
              AND a.status IN ('classified','entity_processed','processed')
              AND (a.category != 'international' 
                   OR a.party_mentioned NOT IN ('[]','') 
                   OR a.ministers_mentioned NOT IN ('[]','') 
                   OR a.states_mentioned NOT IN ('[]','') 
                   OR a.cities_mentioned NOT IN ('[]',''))
              AND (a.ministers_mentioned != '[]' OR a.party_mentioned != '[]' OR a.civic_flag = 1)
            ORDER BY a.id ASC
            LIMIT ?
        """, (start_id, batch_size))
        articles_rows = cursor.fetchall()
    except Exception as e:
        logging.critical(f"Failed to fetch articles batch: {e}")
        conn.close()
        sys.exit(1)

    if not articles_rows:
        logging.info("No articles to process in this batch.")
        print("has_more=false")
        print("articles_attached=0")
        conn.close()
        sys.exit(0)

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
            title_prompt_template = read_prompt('event_title')
            closure_audit_prompt_template = read_prompt('closure_audit')
            saga_check_prompt_template = read_prompt('saga_check')
        except Exception as e:
            logging.critical(f"Failed to load prompt templates: {e}")
            conn.close()
            sys.exit(1)

    # 4. Load Open Events into Memory
    open_events = []
    try:
        cursor.execute("SELECT id, title, entity_keys, centroid, last_seen, article_count, first_seen, scope FROM events WHERE state = 'open'")
        event_rows = cursor.fetchall()
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
                    'scope': r[7]
                })
            except Exception as ex:
                logging.error(f"Error parsing event row {r[0]}: {ex}")
        logging.info(f"Loaded {len(open_events)} open events into memory index.")
    except Exception as e:
        logging.critical(f"Failed to load open events: {e}")
        conn.close()
        sys.exit(1)

    attached_count = 0
    last_processed_id = start_id
    max_scraped_at = 0

    # 5. Pipeline Matching Loop
    for row in articles_rows:
        art_id = int(row[0])
        orig_title = row[1] or ""
        reph_title = row[2] or orig_title
        decompressed_article = decompress_text(row[3])
        scraped_at = int(row[4]) if row[4] is not None else 0
        max_scraped_at = max(max_scraped_at, scraped_at)
        last_processed_id = art_id

        # Parse entities
        art_entities = extract_keys(row[5], row[6], row[7], row[8])

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

            # Step 2: Filter candidates in memory (candidates only if state='open' and first_seen > scraped_at - 21d)
            cutoff_time = scraped_at - 21 * 24 * 3600
            for ev in open_events:
                if ev['first_seen'] > cutoff_time:
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

                    # Invoke Gemma 9B for Vote
                    output = llm_9b(prompt, max_tokens=150, stop=["<end_of_turn>"], temperature=0.0)
                    response_text = output['choices'][0]['text'].strip()
                    vote = "ATTACH" if response_text.upper().startswith("ATTACH") else "REJECT"

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
                    # Write Milestone using Gemma 2B
                    prompt = milestone_prompt_template.format(
                        title=orig_title,
                        summary=decompressed_article[:1200]
                    )
                    output = llm_2b(prompt, max_tokens=100, stop=["<end_of_turn>"], temperature=0.1)
                    gen_milestone = output['choices'][0]['text'].strip()

                    # Validate milestone (max 30 words, no quotes)
                    words = gen_milestone.split()
                    valid = True
                    if len(words) < 3 or len(words) > 30:
                        valid = False
                    if gen_milestone.startswith('"') or gen_milestone.endswith('"') or gen_milestone.startswith("'") or gen_milestone.endswith("'"):
                        valid = False
                    
                    if valid:
                        milestone = gen_milestone
                        logging.info(f"  Generated Milestone: '{milestone}'")
                    else:
                        logging.info(f"  Milestone validation failed (length={len(words)}, text='{gen_milestone}'). Fallback to NULL.")

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
                        title_out = llm_9b(title_prompt, max_tokens=60, stop=["<end_of_turn>"], temperature=0.0)
                        gen_title = title_out['choices'][0]['text'].strip().strip('"').strip("'")
                        
                        # Validate title length
                        if len(gen_title.split()) <= 8:
                            event_title = gen_title
                            event_slug = slugify(event_title)
                            logging.info(f"  Generated Event Title: '{event_title}' | slug: {event_slug}")
                        else:
                            logging.error(f"  Generated Title too long ({len(gen_title.split())} words): '{gen_title}'. Keeping NULL.")

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
                        art_id
                    )
                    cursor = conn.cursor()
                except Exception as commit_err:
                    logging.critical(f"Failed to advance cursor after poison-pill: {commit_err}")
                    conn.close()
                    sys.exit(1)
            continue

    # 6. Chapter Closure Sweep (Logical Clock)
    # state='open' AND first_seen < batch_max_scraped_at - 21d -> run CLOSURE CEREMONY -> set state='closed'
    if not args.dry_run and max_scraped_at > 0:
        closure_cutoff = max_scraped_at - 21 * 24 * 3600
        logging.info(f"\nRunning chapter closure sweep relative to logical clock cutoff: {closure_cutoff}...")
        
        # 1. One-off migration of dormant events to closed state
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
                
        # 2. Find all open events that need to close
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM events WHERE state = 'open' AND first_seen < ?", (closure_cutoff,))
            events_to_close = [r[0] for r in cursor.fetchall()]
            
            if events_to_close:
                logging.info(f"Found {len(events_to_close)} events to close.")
                for ev_id in events_to_close:
                    logging.info(f"Closing Event ID {ev_id}...")
                    try:
                        # Refresh cursor for reads
                        cursor = conn.cursor()
                        # Phase 1: All LLM decisions gathered in memory (No database writes during this phase)
                        evicted_ids, linked_ids = get_closure_decisions(
                            cursor, llm_9b, ev_id,
                            closure_audit_prompt_template, saga_check_prompt_template
                        )
                        
                        # Phase 2: Apply all changes in one single write transaction burst
                        conn, _ = run_write_burst_with_reconnect(
                            conn,
                            do_closure_write,
                            encoder,
                            ev_id,
                            evicted_ids,
                            linked_ids
                        )
                        cursor = conn.cursor()
                        logging.info(f"Event ID {ev_id} closed successfully.")
                    except Exception as e:
                        logging.error(f"Failed to run closure ceremony for Event ID {ev_id}: {e}")
        except Exception as e:
            logging.error(f"Failed to query events for closure sweep: {e}")

    # Check if there are more articles remaining in the table (with reconnect logic)
    has_more = "false"
    for attempt in range(2):
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM articles 
                WHERE id > ?
                  AND status IN ('classified','entity_processed','processed')
                  AND (category != 'international' 
                       OR party_mentioned NOT IN ('[]','') 
                       OR ministers_mentioned NOT IN ('[]','') 
                       OR states_mentioned NOT IN ('[]','') 
                       OR cities_mentioned NOT IN ('[]',''))
                  AND (ministers_mentioned != '[]' OR party_mentioned != '[]' OR civic_flag = 1)
                LIMIT 1
            """, (last_processed_id,))
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

if __name__ == '__main__':
    main()
