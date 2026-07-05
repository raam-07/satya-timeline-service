import os
import json
import re
import sys
import logging
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_eval():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(script_dir, "eval", "eval_cases.jsonl")
    prompt_path = os.path.join(script_dir, "prompts", "attach_gate.txt")

    if not os.path.exists(cases_path):
        print(f"Error: Evaluation cases not found at {cases_path}")
        sys.exit(1)
    if not os.path.exists(prompt_path):
        print(f"Error: Attach gate prompt not found at {prompt_path}")
        sys.exit(1)

    # Read prompt template
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Load 9B model
    model_9b_path = os.environ.get('MODEL_9B_PATH')
    if not model_9b_path:
        # Check standard locations
        possible_paths = [
            os.path.join(script_dir, "models", "gemma-2-9b-it-Q6_K.gguf"),
            os.path.join(os.path.dirname(script_dir), "models", "gemma-2-9b-it-Q6_K.gguf"),
            "./models/gemma-2-9b-it-Q6_K.gguf"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                model_9b_path = p
                break
    
    if not model_9b_path or not os.path.exists(model_9b_path):
        print(f"Error: Gemma 9B model not found at {model_9b_path or 'any standard location'}")
        sys.exit(1)

    print(f"Loading Gemma 9B model from: {model_9b_path}...")
    llm_9b = Llama(model_path=model_9b_path, n_ctx=2048, verbose=False)
    print("Model loaded successfully.")

    # Read cases
    cases = []
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))

    print(f"Loaded {len(cases)} evaluation cases.")

    results = []
    
    # Process all cases
    for idx, c in enumerate(cases):
        event_title = c["event_title"]
        scope = c["scope"]
        recent_milestones = c["recent_milestones"]
        article_title = c["article_title"]
        article_summary = c["article_summary"]
        expected = c["expected"]
        held_out = c.get("held_out", False)

        prompt = prompt_template.format(
            event_title=event_title,
            scope=scope,
            recent_milestones=recent_milestones,
            new_title=article_title,
            new_summary=article_summary[:800]
        )

        output = llm_9b(prompt, max_tokens=150, stop=["<end_of_turn>"], temperature=0.0)
        response_text = output['choices'][0]['text'].strip()
        predicted = "ATTACH" if response_text.upper().startswith("ATTACH") else "REJECT"

        is_correct = (predicted == expected)
        is_junk_admit = (expected == "REJECT" and predicted == "ATTACH")

        results.append({
            "case": c,
            "predicted": predicted,
            "is_correct": is_correct,
            "is_junk_admit": is_junk_admit,
            "held_out": held_out
        })

        status = "✓ CORRECT" if is_correct else "✗ FAIL"
        if is_junk_admit:
            status += " (JUNK ADMIT VIOLATION)"
        print(f"[{idx+1}/{len(cases)}] Event: {event_title} | Article: {article_title[:40]}... | Expected: {expected} | Predicted: {predicted} -> {status}")

    # Compute metrics
    total_tune = 0
    correct_tune = 0
    junk_admits_tune = 0

    total_test = 0
    correct_test = 0
    junk_admits_test = 0

    misclassifications = []

    for r in results:
        is_correct = r["is_correct"]
        is_junk = r["is_junk_admit"]
        held_out = r["held_out"]
        c = r["case"]
        pred = r["predicted"]
        exp = c["expected"]

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
    total_cases = total_tune + total_test
    total_correct = correct_tune + correct_test
    acc_overall = (total_correct / total_cases) * 100 if total_cases > 0 else 0
    total_junk_admits = junk_admits_tune + junk_admits_test

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Tuning set (80%): {correct_tune}/{total_tune} correct ({acc_tune:.2f}% accuracy)")
    print(f"Tuning set junk-admits: {junk_admits_tune}")
    print(f"Held-out set (20%): {correct_test}/{total_test} correct ({acc_test:.2f}% accuracy)")
    print(f"Held-out set junk-admits: {junk_admits_test}")
    print(f"Overall accuracy: {total_correct}/{total_cases} correct ({acc_overall:.2f}%)")
    print(f"Total junk-admits overall: {total_junk_admits}")
    print("====================================================")

    if misclassifications:
        print("\n--- MISCLASSIFICATIONS ---")
        for idx, m in enumerate(misclassifications):
            test_tag = "[HELD-OUT]" if m["held_out"] else "[TUNE]"
            junk_tag = " (JUNK ADMIT!)" if m["is_junk"] else ""
            print(f"{idx+1}. {test_tag} Event: {m['event']} | Article: {m['article']} | Expected: {m['expected']} | Predicted: {m['predicted']}{junk_tag}")

    # Validation criteria checks
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
        sys.exit(0)
    else:
        print("\nOverall Status: FAIL! Validation criteria not met.")
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
