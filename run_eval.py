import os
import json
import re
import sys
import time
import traceback
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def summarize(output_path, total_cases):
    """Read an eval output file (one JSON record per case) and print/return
    the pass-fail summary. Used both at the end of a normal run and standalone
    to merge/summarize shard outputs without reloading the model."""
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    total_tune = 0
    correct_tune = 0
    junk_admits_tune = 0

    total_test = 0
    correct_test = 0
    junk_admits_test = 0

    misclassifications = []
    technical_errors = []

    for r in results:
        is_correct = r["is_correct"]
        is_junk = r["is_junk_admit"]
        held_out = r["held_out"]
        c = r["case"]
        pred = r["predicted"]
        exp = c["expected"]

        if r.get("error"):
            technical_errors.append({
                "event": c["event_title"],
                "article": c["article_title"],
                "error": r["error"],
                "stage": r.get("stage")
            })

        if not is_correct:
            misclassifications.append({
                "event": c["event_title"],
                "article": c["article_title"],
                "expected": exp,
                "predicted": pred,
                "held_out": held_out,
                "is_junk": is_junk
            })

        if held_out:
            total_test += 1
            if is_correct:
                correct_test += 1
            if is_junk:
                junk_admits_test += 1
        else:
            total_tune += 1
            if is_correct:
                correct_tune += 1
            if is_junk:
                junk_admits_tune += 1

    acc_tune = (correct_tune / total_tune) * 100 if total_tune > 0 else 0
    acc_test = (correct_test / total_test) * 100 if total_test > 0 else 0
    total_cases_seen = total_tune + total_test
    total_correct = correct_tune + correct_test
    acc_overall = (total_correct / total_cases_seen) * 100 if total_cases_seen > 0 else 0
    total_junk_admits = junk_admits_tune + junk_admits_test

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Cases recorded in {output_path}: {total_cases_seen}/{total_cases}")
    if total_cases_seen < total_cases:
        print(f"NOTE: incomplete — {total_cases - total_cases_seen} cases not yet processed.")
    print(f"Tuning set (80%): {correct_tune}/{total_tune} correct ({acc_tune:.2f}% accuracy)")
    print(f"Tuning set junk-admits: {junk_admits_tune}")
    print(f"Held-out set (20%): {correct_test}/{total_test} correct ({acc_test:.2f}% accuracy)")
    print(f"Held-out set junk-admits: {junk_admits_test}")
    print(f"Overall accuracy (of cases seen): {total_correct}/{total_cases_seen} correct ({acc_overall:.2f}%)")
    print(f"Total junk-admits overall: {total_junk_admits}")
    print(f"Technical errors (LLM call failed, defaulted to REJECT): {len(technical_errors)}")
    print("====================================================")

    if technical_errors:
        print("\n--- TECHNICAL ERRORS (not model judgment — investigate separately) ---")
        for idx, e in enumerate(technical_errors):
            print(f"{idx+1}. Event: {e['event']} | Article: {e['article']} | Stage: {e['stage']} | Error: {e['error']}")

    if misclassifications:
        print("\n--- MISCLASSIFICATIONS ---")
        for idx, m in enumerate(misclassifications):
            test_tag = "[HELD-OUT]" if m["held_out"] else "[TUNE]"
            junk_tag = " (JUNK ADMIT!)" if m["is_junk"] else ""
            print(f"{idx+1}. {test_tag} Event: {m['event']} | Article: {m['article']} | Expected: {m['expected']} | Predicted: {m['predicted']}{junk_tag}")

    if total_cases_seen < total_cases:
        print("\nOverall Status: INCOMPLETE — cannot judge pass/fail until all cases are processed.")
        return 2

    passed = True
    print("\n--- CRITERIA CHECKS ---")
    if acc_overall < 93.0:
        print(f"FAIL: Overall accuracy {acc_overall:.2f}% is below pass bar of 93.0%")
        passed = False
    else:
        print(f"PASS: Overall accuracy {acc_overall:.2f}% is >= 93.0%")

    if total_junk_admits > 0:
        print(f"FAIL: Found {total_junk_admits} junk-admit violations (no REJECT case may get ATTACH)")
        passed = False
    else:
        print("PASS: 0 junk-admit violations")

    if passed:
        print("\nOverall Status: PASS! All criteria met.")
        return 0
    else:
        print("\nOverall Status: FAIL! Validation criteria not met.")
        return 1


def run_eval(start_index=0, limit=None, output_path=None):
    from llama_cpp import Llama  # deferred: --summarize-only never needs this

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(script_dir, "eval", "eval_cases.jsonl")
    prompt_path = os.path.join(script_dir, "prompts", "attach_gate.txt")
    if output_path is None:
        output_path = os.path.join(script_dir, "eval", "eval_run_output.jsonl")

    if not os.path.exists(cases_path):
        print(f"Error: Evaluation cases not found at {cases_path}")
        sys.exit(1)
    if not os.path.exists(prompt_path):
        print(f"Error: Attach gate prompt not found at {prompt_path}")
        sys.exit(1)

    # Read prompt template
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Load Gate model (Qwen 14B)
    model_9b_path = os.environ.get('MODEL_GATE_PATH') or os.environ.get('MODEL_9B_PATH')
    if not model_9b_path:
        # Check standard locations
        possible_paths = [
            os.path.join(script_dir, "models", "Qwen2.5-14B-Instruct-Q6_K.gguf"),
            os.path.join(os.path.dirname(script_dir), "models", "Qwen2.5-14B-Instruct-Q6_K.gguf"),
            "./models/Qwen2.5-14B-Instruct-Q6_K.gguf"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_9b_path = p
                break

    if not model_9b_path or not os.path.exists(model_9b_path):
        print(f"Error: Gate model not found at {model_9b_path or 'any standard location'}")
        sys.exit(1)

    print(f"Loading Qwen 14B model from: {model_9b_path}...")
    try:
        llm_9b = Llama(model_path=model_9b_path, n_ctx=2048, verbose=False)
    except Exception:
        print("\n=== FATAL: model failed to load ===")
        traceback.print_exc()
        sys.exit(3)
    print("Qwen 14B model loaded successfully.")

    # Read cases
    cases = []
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))

    print(f"Loaded {len(cases)} evaluation cases.")

    end_index = len(cases) if limit is None else min(len(cases), start_index + limit)
    slice_to_run = list(range(start_index, end_index))
    print(f"Processing cases [{start_index+1}..{end_index}] of {len(cases)} (resumable, incremental output at {output_path}).")

    # Fresh run (start_index == 0) truncates the output file; a resumed run
    # (start_index > 0) appends, so a timeout never loses prior progress.
    file_mode = "w" if start_index == 0 else "a"
    out_f = open(output_path, file_mode, encoding="utf-8")

    try:
        for idx in slice_to_run:
            c = cases[idx]
            event_title = c["event_title"]
            scope = c["scope"]
            recent_milestones = c["recent_milestones"]
            article_title = c["article_title"]
            article_summary = c["article_summary"]
            expected = c["expected"]
            held_out = c.get("held_out", False)

            event_began = c.get("event_began", "N/A")
            article_date = c.get("article_date", "N/A")

            response_text = ""
            error = None
            elapsed = 0.0

            if event_began != "N/A" and article_date != "N/A" and article_date < event_began:
                predicted = "REJECT"
                stage = "date-guard"
                print(f"[{idx+1}/{len(cases)}] Date guard triggered (article {article_date} < event {event_began}). Skipping LLM call.")
            else:
                prompt = prompt_template.format(
                    event_title=event_title,
                    scope=scope,
                    event_began=event_began,
                    recent_milestones=recent_milestones,
                    article_date=article_date,
                    new_title=article_title,
                    new_summary=article_summary[:800]
                )

                t0 = time.time()
                try:
                    output = llm_9b(prompt, max_tokens=350, stop=["<|im_end|>"], temperature=0.0)
                    response_text = output['choices'][0]['text'].strip()
                    # Reasoning-first prompt: the verdict is the LAST
                    # ATTACH/REJECT token in the response, not the first word.
                    verdict_matches = re.findall(r'\b(ATTACH|REJECT)\b', response_text.upper())
                    predicted = verdict_matches[-1] if verdict_matches else "REJECT"
                    stage = "llm-gate"
                    if not verdict_matches:
                        error = "no ATTACH/REJECT token found in model output — defaulted to REJECT"
                except Exception as e:
                    # One bad case must not kill the whole run. Log it, mark
                    # it clearly as a technical failure (not a model
                    # judgment), fail-safe to REJECT, and keep going.
                    predicted = "REJECT"
                    stage = "llm-gate-error"
                    error = f"{type(e).__name__}: {e}"
                    logging.error(f"[{idx+1}/{len(cases)}] LLM call raised an exception for '{article_title}': {error}")
                elapsed = time.time() - t0

            is_correct = (predicted == expected)
            is_junk_admit = (expected == "REJECT" and predicted == "ATTACH")

            record = {
                "case": c,
                "predicted": predicted,
                "is_correct": is_correct,
                "is_junk_admit": is_junk_admit,
                "held_out": held_out,
                "stage": stage,
                "raw_response": response_text,
                "error": error,
                "elapsed_seconds": round(elapsed, 2)
            }

            # Write + flush immediately so a timeout/cancellation never loses
            # a completed case — this is the durable record of progress.
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            status = "✓ CORRECT" if is_correct else "✗ FAIL"
            if is_junk_admit:
                status += " (JUNK ADMIT VIOLATION)"
            if error:
                status += f" [WARN: {error}]"
            print(f"[{idx+1}/{len(cases)}] ({elapsed:.1f}s) Event: {event_title} | Article: {article_title[:40]}... | Expected: {expected} | Predicted: {predicted} | Stage: {stage} -> {status}")
    finally:
        out_f.close()

    # For a sharded/partial run, only THIS shard's slice is meaningful in
    # output_path (it's a shard-specific file) — so only print the full
    # pass/fail summary when this run covers the whole case set itself.
    # (The merge step calls summarize() separately over the combined file.)
    if start_index == 0 and (limit is None or limit >= len(cases)):
        exit_code = summarize(output_path, len(cases))
        sys.exit(exit_code)
    else:
        print(f"\nShard/chunk done: cases [{start_index+1}..{end_index}] written to {output_path}.")
        print("Run with --summarize-only against the merged/full output file to get the pass/fail verdict.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-index', type=int, default=0, help="0-based case index to start/resume from")
    parser.add_argument('--limit', type=int, default=None, help="Number of cases to process this run (default: all remaining)")
    parser.add_argument('--output-path', type=str, default=None, help="Path to write/append incremental results (default: eval/eval_run_output.jsonl)")
    parser.add_argument('--summarize-only', action='store_true', help="Skip model loading/inference — just summarize an existing output file")
    args = parser.parse_args()

    if args.summarize_only:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cases_path = os.path.join(script_dir, "eval", "eval_cases.jsonl")
        total_cases = sum(1 for line in open(cases_path, encoding="utf-8") if line.strip())
        output_path = args.output_path or os.path.join(script_dir, "eval", "eval_run_output.jsonl")
        sys.exit(summarize(output_path, total_cases))
    else:
        run_eval(start_index=args.start_index, limit=args.limit, output_path=args.output_path)
