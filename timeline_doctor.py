"""
timeline_doctor.py — surgical repair tool for individual articles' timelines.

Two-phase design so many runners can think in parallel while exactly one writes:

  JUDGE (read-only, runs on up to 18 runners in parallel):
    --judge --ids 123,456   (or --ids-file slice.txt)
    For each article: eligibility check -> already attached? -> else find the
    best candidate event (any state, overlapping the article's OWN date window)
    -> Qwen gate verdict -> emit a decision to decisions_<tag>.jsonl.
    With --check-siblings (auto when <=5 ids): for attached articles, hunt the
    event's window for eligible articles that SHOULD belong but don't, gate
    each, and emit attach decisions for approved ones.

  APPLY (single runner, serial writes in scraped_at order):
    --apply --decisions-dir ./decisions
    Merges all decision files, de-duplicates, applies attaches/foundings with
    the same write discipline as the pipeline. NEVER touches the forward
    checkpoint. A 'found' decision is downgraded to attach if an event founded
    earlier in this same pass matches by keys+cosine (prevents parallel-judge
    duplicate foundings).
"""
import os
import sys
import json
import base64
import glob
import logging
import argparse
from datetime import datetime

import numpy as np

from timeline_pipeline import (
    get_db_connection,
    load_models,
    read_prompt,
    decompress_text,
    extract_keys,
    generate_event_scope,
    clean_title,
    slugify,
    run_write_burst_with_reconnect,
    ELIGIBILITY_SQL,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SIM_THRESHOLD = 0.60
WINDOW_BEFORE = 2 * 86400        # candidate events may start up to 2d after the article
WINDOW_AFTER = 21 * 86400        # ...or have been alive within 21d before it


def fetch_articles(conn, ids):
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in ids)
    cursor.execute(f"""
        SELECT a.id, a.title, a.rephrased_title, a.rephrased_article, a.scraped_at,
               a.party_mentioned, a.ministers_mentioned, a.states_mentioned,
               a.cities_mentioned, a.civic_flag,
               (SELECT COUNT(*) FROM articles e WHERE e.id = a.id AND {ELIGIBILITY_SQL.replace('a.', 'e.')}) AS eligible
        FROM articles a WHERE a.id IN ({placeholders})
    """, list(ids))
    return cursor.fetchall()


def fetch_candidate_events(conn, scraped_at):
    """Events (ANY state) whose lifespan overlaps the article's own era."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, entity_keys, centroid, first_seen, last_seen,
               article_count, state, scope
        FROM events
        WHERE first_seen <= ? AND last_seen >= ?
    """, (scraped_at + WINDOW_BEFORE, scraped_at - WINDOW_AFTER))
    out = []
    for r in cursor.fetchall():
        try:
            out.append({
                'id': int(r[0]), 'title': r[1],
                'keys': set(json.loads(r[2])) if r[2] else set(),
                'centroid': np.frombuffer(r[3], dtype=np.float32) if r[3] else None,
                'first_seen': int(r[4] or 0), 'last_seen': int(r[5] or 0),
                'article_count': int(r[6] or 0), 'state': r[7], 'scope': r[8],
            })
        except Exception as e:
            logging.error(f"Unparseable event {r[0]}: {e}")
    return out


def already_attached(conn, article_id):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ea.event_id, e.title, e.article_count, e.state
        FROM event_articles ea JOIN events e ON e.id = ea.event_id
        WHERE ea.article_id = ?
    """, (article_id,))
    return cursor.fetchone()


def recent_milestones(conn, event_id, n=2):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COALESCE(ea.milestone, a.rephrased_title, a.title)
        FROM event_articles ea JOIN articles a ON a.id = ea.article_id
        WHERE ea.event_id = ? ORDER BY ea.event_date DESC LIMIT ?
    """, (event_id, n))
    return [r[0] for r in cursor.fetchall() if r[0]]


def fmt_date(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "N/A"


def gate_verdict(llm_9b, gate_tmpl, event, milestones, art_title, art_summary, art_date):
    prompt = gate_tmpl.format(
        event_title=event['title'] or 'Untitled event',
        scope=event['scope'] or (milestones[0] if milestones else 'No scope recorded'),
        event_began=fmt_date(event['first_seen']),
        recent_milestones="\n".join(f"- {m}" for m in milestones) or "- (none)",
        article_date=fmt_date(art_date),
        new_title=art_title,
        new_summary=art_summary[:1200],
    )
    out = llm_9b(prompt, max_tokens=400, stop=["<|im_end|>"], temperature=0.0)
    text = out['choices'][0]['text'].strip()
    verdict = 'ATTACH' if text.upper().rstrip().endswith('ATTACH') else 'REJECT'
    return verdict, text[-300:]


def gen_milestone(llm_2b, milestone_tmpl, title, summary):
    try:
        out = llm_2b(milestone_tmpl.format(title=title, summary=summary[:1200]),
                     max_tokens=80, stop=["<end_of_turn>"], temperature=0.0)
        m = out['choices'][0]['text'].strip().strip('"')
        return m if 0 < len(m.split()) <= 32 else None
    except Exception as e:
        logging.error(f"Milestone generation failed: {e}")
        return None


def judge(args):
    ids = []
    if args.ids:
        ids = [int(x) for x in args.ids.split(',') if x.strip()]
    if args.ids_file:
        with open(args.ids_file) as f:
            ids += [int(x) for x in f.read().replace(',', '\n').split() if x.strip()]
    if not ids:
        logging.critical("No article ids given.")
        sys.exit(1)

    conn = get_db_connection()
    encoder, llm_2b, llm_9b = load_models()
    gate_tmpl = read_prompt('attach_gate')
    milestone_tmpl = read_prompt('milestone')
    scope_tmpl = read_prompt('event_scope')

    check_siblings = args.check_siblings or len(ids) <= 5
    decisions = []

    rows = fetch_articles(conn, ids)
    found_ids = {int(r[0]) for r in rows}
    for missing in set(ids) - found_ids:
        decisions.append({'article_id': missing, 'action': 'skip', 'reason': 'article id not found in DB'})

    for r in rows:
        art_id = int(r[0])
        title = r[2] or r[1] or ''
        summary = decompress_text(r[3])
        scraped_at = int(r[4] or 0)
        eligible = bool(r[10])
        logging.info(f"\n=== Article {art_id}: '{title[:70]}'")

        if not eligible:
            decisions.append({'article_id': art_id, 'action': 'skip',
                              'reason': 'not timeline-eligible (entities/category filter)'})
            logging.info("  -> skip: not eligible")
            continue

        attached = already_attached(conn, art_id)
        if attached:
            ev_id, ev_title, ev_count, ev_state = attached
            logging.info(f"  -> already in event {ev_id} '{ev_title}' ({ev_count} articles, {ev_state})")
            decisions.append({'article_id': art_id, 'action': 'report',
                              'reason': f"already in event {ev_id} '{ev_title}' ({ev_count} articles, {ev_state})"})
            if check_siblings:
                decisions += hunt_siblings(conn, encoder, llm_9b, llm_2b, gate_tmpl, milestone_tmpl, ev_id)
            continue

        # Not attached: judge against historical candidates
        emb = encoder.encode(f"{title} {summary}".strip(), convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        entities = extract_keys(r[5], r[6], r[7], r[8])

        candidates = fetch_candidate_events(conn, scraped_at)
        best, best_sim = None, -1.0
        for ev in candidates:
            if ev['centroid'] is None or len(ev['centroid']) != len(emb):
                continue
            if not ev['keys'].intersection(entities):
                continue
            sim = float(np.dot(ev['centroid'], emb))
            if sim > best_sim:
                best, best_sim = ev, sim
        logging.info(f"  best candidate: {best['id'] if best else None} (cos {best_sim:.3f})")

        if best and best_sim >= SIM_THRESHOLD:
            ms = recent_milestones(conn, best['id'])
            verdict, raw = gate_verdict(llm_9b, gate_tmpl, best, ms, title, summary, scraped_at)
            logging.info(f"  gate: {verdict}")
            if verdict == 'ATTACH':
                decisions.append({
                    'article_id': art_id, 'action': 'attach', 'event_id': best['id'],
                    'milestone': gen_milestone(llm_2b, milestone_tmpl, title, summary),
                    'scraped_at': scraped_at, 'entities': list(entities),
                    'embedding_b64': base64.b64encode(emb.astype(np.float32).tobytes()).decode(),
                    'reason': f"gate ATTACH to event {best['id']} (cos {best_sim:.3f})",
                })
                continue

        # Found a new event for it
        scope = generate_event_scope(llm_9b, scope_tmpl, title, summary)
        decisions.append({
            'article_id': art_id, 'action': 'found',
            'scope': scope, 'scraped_at': scraped_at, 'entities': list(entities),
            'embedding_b64': base64.b64encode(emb.astype(np.float32).tobytes()).decode(),
            'milestone': None,
            'reason': f"no candidate passed (best cos {best_sim:.3f})",
        })
        logging.info("  -> will found new event")

    out_path = args.out or f"decisions_{args.tag or 'run'}.jsonl"
    with open(out_path, 'w') as f:
        for d in decisions:
            f.write(json.dumps(d) + '\n')
    counts = {}
    for d in decisions:
        counts[d['action']] = counts.get(d['action'], 0) + 1
    logging.info(f"\nJudge done: {counts} -> {out_path}")
    conn.close()


def hunt_siblings(conn, encoder, llm_9b, llm_2b, gate_tmpl, milestone_tmpl, event_id):
    """Eligible articles inside the event's window, sharing keys, close by
    cosine, but NOT attached to any event — gate each, propose attaches."""
    cursor = conn.cursor()
    cursor.execute("SELECT title, entity_keys, centroid, first_seen, last_seen, scope, article_count, state FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    if not row:
        return []
    ev = {'id': event_id, 'title': row[0], 'keys': set(json.loads(row[1]) or []),
          'centroid': np.frombuffer(row[2], dtype=np.float32) if row[2] else None,
          'first_seen': int(row[3] or 0), 'last_seen': int(row[4] or 0), 'scope': row[5]}
    if ev['centroid'] is None:
        return []

    cursor.execute(f"""
        SELECT a.id, a.title, a.rephrased_title, a.rephrased_article, a.scraped_at,
               a.party_mentioned, a.ministers_mentioned, a.states_mentioned, a.cities_mentioned
        FROM articles a
        WHERE a.scraped_at BETWEEN ? AND ?
          AND {ELIGIBILITY_SQL}
          AND a.id NOT IN (SELECT article_id FROM event_articles)
        LIMIT 200
    """, (ev['first_seen'] - WINDOW_BEFORE, ev['last_seen'] + WINDOW_AFTER))
    ms = recent_milestones(conn, event_id)
    out = []
    checked = 0
    for r in cursor.fetchall():
        entities = extract_keys(r[5], r[6], r[7], r[8])
        if not ev['keys'].intersection(entities):
            continue
        title = r[2] or r[1] or ''
        summary = decompress_text(r[3])
        emb = encoder.encode(f"{title} {summary}".strip(), convert_to_numpy=True)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        sim = float(np.dot(ev['centroid'], emb))
        if sim < SIM_THRESHOLD:
            continue
        if checked >= 15:  # bound the LLM bill per event
            logging.info("  sibling hunt: 15-check cap reached")
            break
        checked += 1
        verdict, _raw = gate_verdict(llm_9b, gate_tmpl, ev, ms, title, summary, int(r[4] or 0))
        logging.info(f"  sibling {r[0]} '{title[:50]}' cos {sim:.3f} -> {verdict}")
        if verdict == 'ATTACH':
            out.append({
                'article_id': int(r[0]), 'action': 'attach', 'event_id': event_id,
                'milestone': gen_milestone(llm_2b, milestone_tmpl, title, summary),
                'scraped_at': int(r[4] or 0), 'entities': list(entities),
                'embedding_b64': base64.b64encode(emb.astype(np.float32).tobytes()).decode(),
                'reason': f"missing sibling of event {event_id} (cos {sim:.3f})",
            })
    return out


def _apply_attach(cursor, d, ev_row):
    ev_id = d['event_id']
    count, centroid_blob, keys_json, last_seen = ev_row
    cursor.execute("INSERT INTO event_articles (event_id, article_id, milestone, event_date) VALUES (?, ?, ?, ?)",
                   (ev_id, d['article_id'], d.get('milestone'), d['scraped_at']))
    emb = np.frombuffer(base64.b64decode(d['embedding_b64']), dtype=np.float32)
    old = np.frombuffer(centroid_blob, dtype=np.float32) if centroid_blob else emb
    new_centroid = (old * count + emb) / (count + 1)
    n = np.linalg.norm(new_centroid)
    if n > 0:
        new_centroid = new_centroid / n
    keys = list(dict.fromkeys((json.loads(keys_json) if keys_json else []) + d.get('entities', [])))[:15]
    cursor.execute("""
        UPDATE events SET article_count = ?, centroid = ?, entity_keys = ?, last_seen = MAX(last_seen, ?)
        WHERE id = ?
    """, (count + 1, new_centroid.astype(np.float32).tobytes(), json.dumps(keys), d['scraped_at'], ev_id))


def apply(args):
    files = sorted(glob.glob(os.path.join(args.decisions_dir, '**', '*.jsonl'), recursive=True))
    if not files:
        logging.critical(f"No decision files under {args.decisions_dir}")
        sys.exit(1)
    decisions, seen = [], set()
    for fp in files:
        for line in open(fp):
            d = json.loads(line)
            if d['action'] in ('attach', 'found') and d['article_id'] not in seen:
                seen.add(d['article_id'])
                decisions.append(d)
    decisions.sort(key=lambda d: d.get('scraped_at', 0))
    logging.info(f"Applying {len(decisions)} decisions from {len(files)} files (chronological)...")

    conn = get_db_connection()
    now = int(datetime.now().timestamp())
    founded_this_pass = []  # {'id','keys','centroid'} for duplicate-found downgrade
    applied = {'attach': 0, 'found': 0, 'downgraded': 0, 'skipped': 0}

    for d in decisions:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM event_articles WHERE article_id = ?", (d['article_id'],))
        if cursor.fetchone():
            applied['skipped'] += 1
            continue
        emb = np.frombuffer(base64.b64decode(d['embedding_b64']), dtype=np.float32)

        if d['action'] == 'found':
            # Downgrade to attach if a just-founded event matches (parallel
            # judges can't see each other's foundings).
            ents = set(d.get('entities', []))
            for f in founded_this_pass:
                if f['keys'].intersection(ents) and float(np.dot(f['centroid'], emb)) >= SIM_THRESHOLD:
                    d = {**d, 'action': 'attach', 'event_id': f['id'],
                         'reason': d['reason'] + f" [downgraded: duplicate of pass-founding {f['id']}]"}
                    applied['downgraded'] += 1
                    break

        if d['action'] == 'attach':
            cursor.execute("SELECT article_count, centroid, entity_keys, last_seen FROM events WHERE id = ?", (d['event_id'],))
            ev_row = cursor.fetchone()
            if not ev_row:
                applied['skipped'] += 1
                continue
            conn, _ = run_write_burst_with_reconnect(conn, _apply_attach, d, ev_row)
            applied['attach'] += 1
            logging.info(f"ATTACHED article {d['article_id']} -> event {d['event_id']}")
        else:
            state = 'closed' if (now - d['scraped_at']) > 21 * 86400 else 'open'

            def _found(cursor):
                cursor.execute("""
                    INSERT INTO events (title, slug, entity_keys, centroid, first_seen, last_seen, article_count, state, scope)
                    VALUES (NULL, NULL, ?, ?, ?, ?, 1, ?, ?)
                """, (json.dumps(d.get('entities', [])[:15]), emb.astype(np.float32).tobytes(),
                      d['scraped_at'], d['scraped_at'], state, d.get('scope')))
                cursor.execute("SELECT last_insert_rowid()")
                return cursor.fetchone()[0]

            conn, new_id = run_write_burst_with_reconnect(conn, _found)

            def _member(cursor):
                cursor.execute("INSERT INTO event_articles (event_id, article_id, milestone, event_date) VALUES (?, ?, ?, ?)",
                               (new_id, d['article_id'], d.get('milestone'), d['scraped_at']))
            conn, _ = run_write_burst_with_reconnect(conn, _member)
            founded_this_pass.append({'id': new_id, 'keys': set(d.get('entities', [])), 'centroid': emb})
            applied['found'] += 1
            logging.info(f"FOUNDED event {new_id} ({state}) for article {d['article_id']}")

    logging.info(f"\nApply done: {applied}")
    print(f"doctor_attached={applied['attach']}")
    print(f"doctor_founded={applied['found']}")
    print(f"doctor_downgraded={applied['downgraded']}")
    conn.close()


def main():
    p = argparse.ArgumentParser(description="Timeline Doctor: diagnose & repair timelines per article")
    p.add_argument('--judge', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--ids', type=str, default='')
    p.add_argument('--ids-file', type=str, default='')
    p.add_argument('--check-siblings', action='store_true')
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--out', type=str, default='')
    p.add_argument('--decisions-dir', type=str, default='./decisions')
    args = p.parse_args()
    if args.judge:
        judge(args)
    elif args.apply:
        apply(args)
    else:
        p.error("choose --judge or --apply")


if __name__ == '__main__':
    main()
