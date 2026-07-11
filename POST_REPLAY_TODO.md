# Post-Replay Checklist — do these AFTER all 11 shards are SEALED

## 1. Stitch (Replay 3)
- [ ] Run "Timeline Replay 3 — Stitch and Upload" with dry-run first, check merge numbers look sane
- [ ] Run it for real → wipes old Turso events, uploads rebuilt timelines

## 2. Backfill NULL scopes (one-time cleanup)
Context: during the first hours of the replay (before commit "Normalize Qwen
scope output"), ~20-25% of newly founded events got scope = NULL because the
validator rejected Qwen's numbered-list formatting. These events work (gate
falls back to founding milestone) but judge less precisely.

- [ ] Write/run a one-time script: find `events WHERE scope IS NULL`,
      regenerate scope from each event's founding article (first row in
      event_articles), validate with the (now tolerant) normalizer, UPDATE.
- [ ] Needs: Turso creds + Qwen GGUF. ~2 min per event on CPU.

## 3. Switch on the daily forward loop
- [ ] Uncomment the `schedule:` cron block in `.github/workflows/timeline_pipeline.yml`
- [ ] Verify checkpoint: forward loop must start from the snapshot's `max_id`
      (81076) so it doesn't re-process replayed articles

## 4. Daily capacity decision (parked, but REQUIRED — inflow 400-500/day vs ~250/day capacity)
- [ ] Option A: eval Qwen 7B as gate model (run eval/, need junk_admits = 0) → ~2x speed
- [ ] Option B: hosted free-tier API (e.g. Gemini Flash) for the gate call only → ~10x speed
- [ ] Without one of these the daily loop falls behind permanently

## Notes
- Old timelines stay live on the site during replay; stitch swaps them in one shot
- Entity extractor tags the word "sad" as SAD (Akali Dal) → some foreign junk
  articles (e.g. Trump/Rob Reiner) enter timelines. Upstream classifier fix, low priority.
