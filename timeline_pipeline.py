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

def do_attach_write(cursor, matched_event_id, art_id, milestone, scraped_at, new_centroid, new_count, merged_keys, event_title, event_slug):
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
    
    cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id) VALUES (1, ?)", (art_id,))

def do_create_write(cursor, art_id, art_entities, embedding, scraped_at):
    cursor.execute("""
        INSERT INTO events (title, slug, entity_keys, centroid, first_seen, last_seen, article_count, state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        None,
        None,
        json.dumps(art_entities),
        embedding.tobytes(),
        scraped_at,
        scraped_at,
        1,
        'open'
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

    # 1. Cursor Loading
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
    milestone_prompt_template = ""
    title_prompt_template = ""
    if not args.dry_run:
        try:
            attach_prompt_template = read_prompt('attach_gate')
            milestone_prompt_template = read_prompt('milestone')
            title_prompt_template = read_prompt('event_title')
        except Exception as e:
            logging.critical(f"Failed to load prompt templates: {e}")
            conn.close()
            sys.exit(1)

    # 4. Load Open Events into Memory
    open_events = []
    try:
        cursor.execute("SELECT id, title, entity_keys, centroid, last_seen, article_count FROM events WHERE state = 'open'")
        event_rows = cursor.fetchall()
        for r in event_rows:
            try:
                open_events.append({
                    'id': int(r[0]),
                    'title': r[1],
                    'entity_keys': set(json.loads(r[2])),
                    'centroid': np.frombuffer(r[3], dtype=np.float32),
                    'last_seen': int(r[4]) if r[4] is not None else 0,
                    'article_count': int(r[5])
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

            # Step 2: Filter candidates in memory
            cutoff_time = scraped_at - 90 * 24 * 3600
            for ev in open_events:
                if ev['last_seen'] >= cutoff_time:
                    shared_keys = ev['entity_keys'].intersection(art_entities)
                    if shared_keys:
                        sim = float(np.dot(ev['centroid'], embedding))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = ev

            logging.info(f"  Best similarity match: {best_sim:.4f} with Event ID {best_match['id'] if best_match else None}")

            # Do all database read operations BEFORE starting any LLM calls
            milestone_summary = "- None"
            existing_milestones = []
            if best_sim >= args.sim_threshold and best_match is not None:
                if not args.dry_run:
                    try:
                        cursor.execute("""
                            SELECT milestone, event_date FROM event_articles 
                            WHERE event_id = ? 
                            ORDER BY event_date DESC LIMIT 3
                        """, (best_match['id'],))
                        ms_rows = cursor.fetchall()
                        recent_ms = [f"- {r[0]}" for r in ms_rows if r[0]]
                        if recent_ms:
                            milestone_summary = "\n".join(recent_ms)
                        
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
                if args.dry_run:
                    logging.info(f"  [DRY-RUN] Simulating Gemma 9B attach gate approval for Event ID {best_match['id']} (sim: {best_sim:.4f})")
                    matched_event = best_match
                else:
                    event_name = best_match['title'] or "Untitled Event"

                    # Format Prompt
                    prompt = attach_prompt_template.format(
                        event_title=event_name,
                        entity_keys=", ".join(best_match['entity_keys']),
                        recent_milestones=milestone_summary,
                        new_title=reph_title,
                        new_summary=decompressed_article[:800]
                    )

                    # Invoke Gemma 9B (Gate)
                    output = llm_9b(prompt, max_tokens=150, stop=["<end_of_turn>"], temperature=0.0)
                    response_text = output['choices'][0]['text'].strip()
                    logging.info(f"  Gemma 9B Response: {response_text}")

                    # Validation
                    if response_text.upper().startswith("ATTACH"):
                        matched_event = best_match
                        logging.info(f"  [ATTACH] Confirmed by Gemma 9B.")
                    else:
                        logging.info(f"  [REJECT] Rejected by Gemma 9B. Reason details: {response_text}")

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
                    matched_event['entity_keys'] = matched_event['entity_keys'].union(art_entities)
                    
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
                    merged_keys = list(matched_event['entity_keys'].union(art_entities))

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
                        event_slug
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
                        'article_count': 1
                    })
                    attached_count += 1
                    logging.info(f"  [DRY-RUN] Simulated create new Event ID {new_ev_id}.")
                else:
                    # Write to database (transactional execution per article using hot reconnect wrapper)
                    conn, new_ev_id = run_write_burst_with_reconnect(
                        conn,
                        do_create_write,
                        art_id,
                        art_entities,
                        embedding,
                        scraped_at
                    )
                    cursor = conn.cursor()

                    # Update memory cache
                    open_events.append({
                        'id': new_ev_id,
                        'title': None,
                        'entity_keys': set(art_entities),
                        'centroid': embedding,
                        'last_seen': scraped_at,
                        'article_count': 1
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

    # 6. Dormancy Sweep (Logical Clock)
    # Mark open events in database as 'dormant' if they haven't seen updates for 90 days relative to max_scraped_at
    if not args.dry_run and max_scraped_at > 0:
        dormancy_cutoff = max_scraped_at - 90 * 24 * 3600
        logging.info(f"\nRunning dormancy sweep relative to logical clock cutoff: {dormancy_cutoff}...")
        for attempt in range(2):
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE events 
                    SET state = 'dormant' 
                    WHERE state = 'open' AND last_seen < ?
                """, (dormancy_cutoff,))
                dormant_count = cursor.rowcount
                conn.commit()
                logging.info(f"Dormancy sweep completed. Marked {dormant_count} events as dormant.")
                break
            except Exception as e:
                logging.error(f"Failed to execute dormancy sweep (attempt {attempt+1}): {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                if attempt == 0:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = get_db_connection()

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
