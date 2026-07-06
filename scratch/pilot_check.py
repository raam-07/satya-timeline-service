import os
import sys
import time
import json
import gc
import shutil
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

def download_gemma():
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    temp_cache = "./models/.hf_cache"
    os.makedirs(temp_cache, exist_ok=True)

    model_9b_path = os.path.join(models_dir, "gemma-2-9b-it-Q6_K.gguf")
    if not os.path.exists(model_9b_path):
        print("Downloading Gemma 9B...")
        hf_hub_download(
            repo_id="bartowski/gemma-2-9b-it-GGUF",
            filename="gemma-2-9b-it-Q6_K.gguf",
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            cache_dir=temp_cache
        )
    return model_9b_path

def download_qwen():
    qwen_dir = "./qwen_models"
    os.makedirs(qwen_dir, exist_ok=True)
    temp_cache = "./qwen_models/.hf_cache"
    os.makedirs(temp_cache, exist_ok=True)

    model_qwen_path = os.path.join(qwen_dir, "Qwen2.5-14B-Instruct-Q5_K_M.gguf")
    if not os.path.exists(model_qwen_path):
        print("Downloading Qwen 14B...")
        hf_hub_download(
            repo_id="bartowski/Qwen2.5-14B-Instruct-GGUF",
            filename="Qwen2.5-14B-Instruct-Q5_K_M.gguf",
            local_dir=qwen_dir,
            local_dir_use_symlinks=False,
            cache_dir=temp_cache
        )
    # Clean cache dir
    if os.path.exists(temp_cache):
        shutil.rmtree(temp_cache)
    return model_qwen_path

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    prompt_path = os.path.join(project_dir, "prompts", "attach_gate.txt")
    cases_path = os.path.join(project_dir, "eval", "eval_cases.jsonl")

    # Read cases
    cases = []
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    
    # Take first 15 cases
    test_cases = cases[:15]
    print(f"Loaded {len(test_cases)} test cases for pilot check.")

    with open(prompt_path, "r", encoding="utf-8") as f:
        gemma_template = f.read()

    # Convert to Qwen ChatML template
    qwen_template = gemma_template.replace("<start_of_turn>user", "<|im_start|>user")
    qwen_template = qwen_template.replace("<end_of_turn>", "<|im_end|>")
    qwen_template = qwen_template.replace("<start_of_turn>model", "<|im_start|>assistant")

    model_9b_path = download_gemma()

    # --- GEMMA 9B PILOT ---
    print("\n--- Starting Gemma 9B Pilot ---")
    print(f"Loading Gemma 9B from: {model_9b_path}...")
    llm_9b = Llama(model_path=model_9b_path, n_ctx=2048, verbose=False)
    print("Gemma 9B loaded.")

    gemma_times = []
    gemma_tokens = 0
    for idx, c in enumerate(test_cases):
        prompt = gemma_template.format(
            event_title=c["event_title"],
            scope=c["scope"],
            event_began=c.get("event_began", "N/A"),
            recent_milestones=c["recent_milestones"],
            article_date=c.get("article_date", "N/A"),
            new_title=c["article_title"],
            new_summary=c["article_summary"][:800]
        )
        
        t0 = time.time()
        output = llm_9b(prompt, max_tokens=150, stop=["<end_of_turn>"], temperature=0.0)
        dt = time.time() - t0
        gemma_times.append(dt)
        gemma_tokens += output['usage']['completion_tokens']
        response_text = output['choices'][0]['text'].strip()
        print(f"[Gemma {idx+1}/15] time: {dt:.2f}s | tokens: {output['usage']['completion_tokens']} | answer: {response_text}")

    gemma_total_time = sum(gemma_times)
    del llm_9b
    gc.collect()

    # Remove Gemma 9B model file to release 7.6GB of SSD space
    if os.path.exists(model_9b_path):
        print(f"Removing Gemma 9B model file to free disk space: {model_9b_path}")
        os.remove(model_9b_path)
    
    time.sleep(2)

    # Download Qwen 14B to a separate temporary directory
    model_qwen_path = download_qwen()

    # --- QWEN 14B PILOT ---
    print("\n--- Starting Qwen 14B Pilot ---")
    print(f"Loading Qwen 14B from: {model_qwen_path}...")
    llm_qwen = Llama(model_path=model_qwen_path, n_ctx=2048, verbose=False)
    print("Qwen 14B loaded.")

    qwen_times = []
    qwen_tokens = 0
    for idx, c in enumerate(test_cases):
        prompt = qwen_template.format(
            event_title=c["event_title"],
            scope=c["scope"],
            event_began=c.get("event_began", "N/A"),
            recent_milestones=c["recent_milestones"],
            article_date=c.get("article_date", "N/A"),
            new_title=c["article_title"],
            new_summary=c["article_summary"][:800]
        )
        
        t0 = time.time()
        output = llm_qwen(prompt, max_tokens=150, stop=["<|im_end|>"], temperature=0.0)
        dt = time.time() - t0
        qwen_times.append(dt)
        qwen_tokens += output['usage']['completion_tokens']
        response_text = output['choices'][0]['text'].strip()
        print(f"[Qwen {idx+1}/15] time: {dt:.2f}s | tokens: {output['usage']['completion_tokens']} | answer: {response_text}")

    qwen_total_time = sum(qwen_times)
    del llm_qwen
    gc.collect()

    # Clean up Qwen 14B model file and temp folder
    if os.path.exists(model_qwen_path):
        print(f"Cleaning up Qwen 14B file: {model_qwen_path}")
        os.remove(model_qwen_path)
    qwen_dir = os.path.dirname(model_qwen_path)
    if os.path.exists(qwen_dir):
        print(f"Removing temporary Qwen directory: {qwen_dir}")
        shutil.rmtree(qwen_dir)

    # Restore Gemma 9B model file to ensure GHA cache remains complete
    print("\nRestoring Gemma 9B model file to keep cached models folder complete...")
    download_gemma()

    print("\n=== PILOT TIMING RESULTS ===")
    print(f"Gemma 9B Total: {gemma_total_time:.2f}s (avg {gemma_total_time/len(test_cases):.2f}s per call) | Total completion tokens: {gemma_tokens}")
    print(f"Qwen 14B Total: {qwen_total_time:.2f}s (avg {qwen_total_time/len(test_cases):.2f}s per call) | Total completion tokens: {qwen_tokens}")
    diff_pct = ((qwen_total_time - gemma_total_time) / gemma_total_time) * 100
    print(f"Pilot Delta: {diff_pct:+.2f}% wall time change")

if __name__ == "__main__":
    run()
