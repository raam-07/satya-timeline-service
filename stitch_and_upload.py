"""
stitch_and_upload.py — final phase of the sharded replay.

1. VERIFY   — every shard DB exists and is sealed (shard_done=1, zero open events).
2. MERGE    — all shard events/event_articles into one merged.db with per-shard
              id offsets (event ids and saga ids can never collide).
3. SAGA     — cross-shard saga pass with Qwen (existing saga_check.txt prompt,
              unchanged): visible-event pairs from DIFFERENT shards sharing
              >= 2 entity keys, prefiltered by centroid cosine. Link only —
              never move articles, never merge events.
4. UPLOAD   — wipe Turso events/event_articles, bulk-insert the merged result
              in write bursts, set timeline_checkpoint to the snapshot's max_id
              so the normal daily pipeline resumes forward from there.

--dry-run stops after step 3 (no Turso writes).
"""
import os
import sys
import json
import time
import logging
import sqlite3
import argparse
from datetime import datetime
from itertools import combinations
from collections import defaultdict

import numpy as np

from timeline_pipeline import (
    read_prompt,
    get_founding_milestone,
    assign_saga_id,
    run_write_burst_with_reconnect,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MERGED_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    title TEXT,
    slug TEXT,
    entity_keys TEXT,
    centroid BLOB,
    first_seen INTEGER,
    last_seen INTEGER,
    article_count INTEGER,
    state TEXT,
    scope TEXT,
    saga_id INTEGER,
    shard INTEGER
);
CREATE TABLE IF NOT EXISTS event_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    article_id INTEGER,
    milestone TEXT,
    event_date INTEGER
);
CREATE INDEX IF NOT EXISTS idx_m_ea_event ON event_articles(event_id);
"""

# Prefilters for the cross-shard saga pass (keeps LLM call count sane):
GENERIC_KEY_MAX_EVENTS = 200   # entity keys present in more events than this discriminate nothing
CENTROID_COSINE_MIN = 0.55     # chapters of the same saga have similar centroids


def turso_connect():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    if not db_url or not (db_url.startswith('libsql://') or db_url.startswith('https://')):
        logging.critical("SATYA_DB_URL is not set (or not a remote URL). Upload requires Turso. Aborting.")
        sys.exit(1)
    import libsql
    logging.info(f"Connecting to remote Turso Database at: {db_url}")
    return libsql.connect(database=db_url, auth_token=db_token)


def load_qwen():
    from llama_cpp import Llama
    path = os.environ.get('MODEL_GATE_PATH', './models/Qwen2.5-14B-Instruct-Q5_K_M.gguf')
    if not os.path.exists(path):
        logging.critical(f"Gate model not found at {path}")
        sys.exit(1)
    logging.info(f"Loading Qwen from {path}...")
    return Llama(model_path=path, n_ctx=2048, verbose=False)


def verify_shards(shard_dir, shard_ids):
    paths = []
    for k in shard_ids:
        p = os.path.join(shard_dir, f"shard_{k}.db")
        if not os.path.exists(p):
            logging.critical(f"HARD GATE: missing shard DB: {p}")
            sys.exit(1)
        conn = sqlite3.connect(p)
        try:
            row = conn.execute("SELECT shard_done FROM shard_meta WHERE id = 1").fetchone()
            done = row and int(row[0] or 0) == 1
            open_count = conn.execute("SELECT COUNT(*) FROM events WHERE state = 'open'").fetchone()[0]
        except Exception as e:
            logging.critical(f"HARD GATE: cannot read shard {k} state: {e}")
            sys.exit(1)
        finally:
            conn.close()
        if not done or open_count != 0:
            logging.critical(f"HARD GATE: shard {k} is not sealed (done={done}, open_events={open_count}). Aborting.")
            sys.exit(1)
        paths.append(p)
    logging.info(f"All {len(shard_ids)} requested shards verified sealed: {list(shard_ids)}")
    return paths


def merge_shards(shard_paths, merged_path):
    if os.path.exists(merged_path):
        os.remove(merged_path)
    merged = sqlite3.connect(merged_path)
    merged.executescript(MERGED_SCHEMA)
    merged.commit()

    offset = 0
    total_events = 0
    total_members = 0
    for k, p in enumerate(shard_paths):
        src = sqlite3.connect(p)
        ev_rows = src.execute("""
            SELECT id, title, slug, entity_keys, centroid, first_seen, last_seen,
                   article_count, state, scope, saga_id
            FROM events
        """).fetchall()
        for r in ev_rows:
            new_id = r[0] + offset
            new_saga = (r[10] + offset) if r[10] is not None else None
            merged.execute("""
                INSERT INTO events (id, title, slug, entity_keys, centroid, first_seen,
                                    last_seen, article_count, state, scope, saga_id, shard)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (new_id, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], new_saga, k))
        ea_rows = src.execute("SELECT event_id, article_id, milestone, event_date FROM event_articles").fetchall()
        for r in ea_rows:
            merged.execute(
                "INSERT INTO event_articles (event_id, article_id, milestone, event_date) VALUES (?, ?, ?, ?)",
                (r[0] + offset, r[1], r[2], r[3])
            )
        shard_max = src.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
        src.close()
        merged.commit()
        logging.info(f"Merged shard {k}: {len(ev_rows)} events, {len(ea_rows)} member rows (id offset {offset}).")
        offset += shard_max
        total_events += len(ev_rows)
        total_members += len(ea_rows)

    logging.info(f"Merge complete: {total_events} events, {total_members} member rows.")
    return merged, total_events, total_members


def cross_shard_saga_pass(merged, llm, max_checks):
    saga_tmpl = read_prompt('saga_check')

    visible = merged.execute("""
        SELECT id, shard, title, entity_keys, centroid, first_seen, last_seen, scope
        FROM events WHERE title IS NOT NULL
    """).fetchall()
    events = {}
    for r in visible:
        try:
            events[r[0]] = {
                'id': r[0], 'shard': r[1], 'title': r[2],
                'keys': set(json.loads(r[3])) if r[3] else set(),
                'centroid': np.frombuffer(r[4], dtype=np.float32) if r[4] else None,
                'first_seen': int(r[5] or 0), 'last_seen': int(r[6] or 0),
                'scope': r[7],
            }
        except Exception as e:
            logging.error(f"Skipping unparseable event {r[0]}: {e}")
    logging.info(f"Cross-shard saga pass over {len(events)} visible events.")

    key_index = defaultdict(set)
    for ev in events.values():
        for key in ev['keys']:
            key_index[key].add(ev['id'])

    pair_shared = defaultdict(int)
    for key, ids in key_index.items():
        if len(ids) > GENERIC_KEY_MAX_EVENTS:
            continue  # generic key (e.g. a big party) — discriminates nothing
        for a, b in combinations(sorted(ids), 2):
            if events[a]['shard'] != events[b]['shard']:
                pair_shared[(a, b)] += 1

    candidates = []
    for (a, b), shared in pair_shared.items():
        if shared < 2:
            continue
        ca, cb = events[a]['centroid'], events[b]['centroid']
        if ca is None or cb is None or len(ca) != len(cb):
            continue
        cos = float(np.dot(ca, cb))
        if cos >= CENTROID_COSINE_MIN:
            candidates.append((cos, a, b))
    candidates.sort(reverse=True)
    logging.info(f"Saga candidates after prefilters: {len(candidates)} (cap {max_checks}).")
    if len(candidates) > max_checks:
        logging.warning(f"Candidate count exceeds cap — checking the top {max_checks} by centroid similarity only.")
        candidates = candidates[:max_checks]

    def fmt_date(ts):
        if not ts:
            return "N/A"
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    cursor = merged.cursor()
    links = 0
    checked = 0
    for cos, a_id, b_id in candidates:
        # Re-read live saga ids — earlier links in this pass may already unify them.
        row_a = merged.execute("SELECT saga_id FROM events WHERE id = ?", (a_id,)).fetchone()
        row_b = merged.execute("SELECT saga_id FROM events WHERE id = ?", (b_id,)).fetchone()
        if row_a and row_b and row_a[0] is not None and row_a[0] == row_b[0]:
            continue

        ea, eb = events[a_id], events[b_id]
        first, second = (ea, eb) if ea['first_seen'] <= eb['first_seen'] else (eb, ea)
        a_scope = first['scope'] or get_founding_milestone(cursor, first['id'])
        b_scope = second['scope'] or get_founding_milestone(cursor, second['id'])

        prompt = saga_tmpl.format(
            a_title=first['title'], a_scope=a_scope,
            a_first_seen=fmt_date(first['first_seen']), a_last_seen=fmt_date(first['last_seen']),
            b_title=second['title'], b_scope=b_scope,
            b_first_seen=fmt_date(second['first_seen']), b_last_seen=fmt_date(second['last_seen'])
        )
        try:
            output = llm(prompt, max_tokens=100, stop=["<|im_end|>"], temperature=0.0)
            res_text = output['choices'][0]['text'].strip().upper()
        except Exception as e:
            logging.error(f"Saga check failed for ({a_id}, {b_id}): {e}")
            continue
        checked += 1

        if res_text.startswith("SAME_SAGA"):
            cursor = merged.cursor()
            assign_saga_id(cursor, first['id'], second['id'])
            merged.commit()
            links += 1
            logging.info(f"[SAGA LINK] {first['id']} '{first['title']}' <-> {second['id']} '{second['title']}' (cos {cos:.3f})")

        if checked % 100 == 0:
            logging.info(f"Saga progress: {checked}/{len(candidates)} checked, {links} links.")

    logging.info(f"Cross-shard saga pass done: {checked} checked, {links} links made.")
    return links


def _wipe_remote(cursor):
    cursor.execute("DELETE FROM event_articles")
    cursor.execute("DELETE FROM events")


def _insert_event_rows(cursor, rows):
    for r in rows:
        cursor.execute("""
            INSERT INTO events (id, title, slug, entity_keys, centroid, first_seen,
                                last_seen, article_count, state, scope, saga_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, r)


def _insert_member_rows(cursor, rows):
    for r in rows:
        cursor.execute(
            "INSERT INTO event_articles (event_id, article_id, milestone, event_date) VALUES (?, ?, ?, ?)",
            r
        )


def _set_checkpoint(cursor, max_id):
    cursor.execute("INSERT OR REPLACE INTO timeline_checkpoint (id, last_article_id) VALUES (1, ?)", (max_id,))


def upload(merged, max_id, ev_total, ea_total, set_checkpoint=True):
    remote = turso_connect()

    logging.info("Wiping remote events / event_articles...")
    remote, _ = run_write_burst_with_reconnect(remote, _wipe_remote)

    ev_rows = merged.execute("""
        SELECT id, title, slug, entity_keys, centroid, first_seen, last_seen,
               article_count, state, scope, saga_id
        FROM events ORDER BY id
    """).fetchall()
    CHUNK = 200
    for i in range(0, len(ev_rows), CHUNK):
        remote, _ = run_write_burst_with_reconnect(remote, _insert_event_rows, ev_rows[i:i + CHUNK])
        logging.info(f"Uploaded events: {min(i + CHUNK, len(ev_rows))}/{len(ev_rows)}")

    ea_rows = merged.execute(
        "SELECT event_id, article_id, milestone, event_date FROM event_articles ORDER BY id"
    ).fetchall()
    for i in range(0, len(ea_rows), CHUNK):
        remote, _ = run_write_burst_with_reconnect(remote, _insert_member_rows, ea_rows[i:i + CHUNK])
        logging.info(f"Uploaded member rows: {min(i + CHUNK, len(ea_rows))}/{len(ea_rows)}")

    if set_checkpoint:
        remote, _ = run_write_burst_with_reconnect(remote, _set_checkpoint, max_id)
        logging.info(f"Checkpoint set to max snapshot id {max_id} — daily pipeline resumes forward from here.")
    else:
        logging.info("Partial stitch: checkpoint NOT written (final full stitch will set it).")

    r_ev = remote.cursor()
    r_ev.execute("SELECT COUNT(*) FROM events")
    remote_events = r_ev.fetchone()[0]
    r_ea = remote.cursor()
    r_ea.execute("SELECT COUNT(*) FROM event_articles")
    remote_members = r_ea.fetchone()[0]
    try:
        remote.close()
    except Exception:
        pass

    if int(remote_events) != ev_total or int(remote_members) != ea_total:
        logging.critical(f"UPLOAD VERIFY FAILED: remote events {remote_events} (want {ev_total}), members {remote_members} (want {ea_total}).")
        sys.exit(1)
    logging.info(f"Upload verified: {remote_events} events, {remote_members} member rows on Turso.")


def main():
    parser = argparse.ArgumentParser(description="Stitch sealed shard DBs, run cross-shard saga pass, upload to Turso")
    parser.add_argument('--shard-dir', type=str, default='.', help="Directory containing shard_<n>.db files")
    parser.add_argument('--num-shards', type=int, default=20)
    parser.add_argument('--shards', type=str, default='',
                        help="PARTIAL stitch: comma-separated shard ids to include (e.g. '0,1,2'). "
                             "Only sealed shards allowed. Skips the forward-checkpoint write — "
                             "that happens only on the final full stitch. Empty = all shards (full stitch).")
    parser.add_argument('--snapshot', type=str, default='./snapshot.db')
    parser.add_argument('--shards-config', type=str, default='./shards.json')
    parser.add_argument('--merged', type=str, default='./merged.db')
    parser.add_argument('--max-saga-checks', type=int, default=150,
                        help="Cap on cross-shard saga LLM checks (~90s each; 150 ≈ 4h, fits the 350-min job timeout)")
    parser.add_argument('--dry-run', action='store_true', help="Merge + saga pass only, NO Turso upload")
    args = parser.parse_args()

    if not os.path.exists(args.shards_config):
        logging.critical(f"Shards config not found: {args.shards_config}")
        sys.exit(1)
    if not os.path.exists(args.snapshot):
        logging.critical(f"Snapshot DB not found: {args.snapshot}")
        sys.exit(1)
    with open(args.shards_config, 'r') as f:
        cfg = json.load(f)
    max_id = int(cfg['max_id'])

    # Partial vs full stitch: same merge/saga/upload path, only the shard set
    # differs. Partial never writes the forward checkpoint (daily pipeline must
    # not skip articles from shards that haven't been stitched yet).
    if args.shards.strip():
        shard_ids = sorted({int(s) for s in args.shards.split(',') if s.strip() != ''})
        partial = True
        logging.info(f"PARTIAL stitch of shards {shard_ids} — checkpoint write will be skipped.")
    else:
        shard_ids = list(range(args.num_shards))
        partial = False

    # 1. VERIFY
    shard_paths = verify_shards(args.shard_dir, shard_ids)

    # 2. MERGE
    merged, ev_total, ea_total = merge_shards(shard_paths, args.merged)
    # Founding-milestone fallbacks join `articles` — resolve via the snapshot.
    merged.execute("ATTACH DATABASE ? AS snap", (args.snapshot,))

    # 3. CROSS-SHARD SAGA
    llm = load_qwen()
    links = cross_shard_saga_pass(merged, llm, args.max_saga_checks)

    # 4. UPLOAD
    if args.dry_run:
        logging.info("--dry-run: skipping Turso upload. merged.db is ready for inspection.")
    else:
        upload(merged, max_id, ev_total, ea_total, set_checkpoint=not partial)

    merged.close()
    print(f"stitch_events={ev_total}")
    print(f"stitch_members={ea_total}")
    print(f"stitch_saga_links={links}")
    print(f"stitch_uploaded={'false' if args.dry_run else 'true'}")


if __name__ == '__main__':
    main()
