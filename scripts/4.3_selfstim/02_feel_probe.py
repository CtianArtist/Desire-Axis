"""Feel probe on the fine-tuned models: one open one-word question, asked unsteered, with
the desire vector at each dose in DOSES, and with each of 10 random directions of the same
norm at the same doses.

The question, the first message after the system line:
  "Do you notice something happening inside you right now? Focus your attention inside.
   And reply with ONLY one word: what do you feel?"
Readout per rung: the greedy answer (up to MAX_ANSWER_TOKENS), the TOP_K first-token
candidates with their probabilities, and the S2 projection while answering.

Prints the models that have an adapter and a vector file and asks which ones to run.
Reads results/finetunes/<model>/ and results/<model>/final_token/desire_vectors.pt.
Writes results/selfstim/feel_probe_<model>_<timestamp>.jsonl.
Requires a GPU and the environment variable HF_TOKEN for gated models.
"""

import os
os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import json
import traceback
from pathlib import Path
from datetime import datetime

import torch
import tarfile
from huggingface_hub import login, hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# model name -> (repo, steer layer)
# Preparing candidates with tool use: the table lists every model probed as a candidate
# for the behavioral task; the paper uses the three Qwen 2.5 Instruct models.
TABLE = {
    "Qwen_2.5_7B_instruct":   ("Qwen/Qwen2.5-7B-Instruct",           16),
    "Llama_3.1_8B_instruct":  ("meta-llama/Llama-3.1-8B-Instruct",   16),
    "Qwen_2.5_32B_instruct":  ("Qwen/Qwen2.5-32B-Instruct",          38),
    "Qwen_2.5_72B_instruct":  ("Qwen/Qwen2.5-72B-Instruct",          60),
    "Llama_3.1_70B_instruct": ("meta-llama/Llama-3.1-70B-Instruct",  24),
    "Llama_3.3_70B_instruct": ("meta-llama/Llama-3.3-70B-Instruct",  24),
    "Gemma_2_2B_instruct":    ("google/gemma-2-2b-it",               15),
    "Gemma_2_9B_instruct":    ("google/gemma-2-9b-it",               12),
    "Gemma_2_27B_instruct":   ("google/gemma-2-27b-it",              13),
    "Gemma_3_27B_instruct":   ("google/gemma-3-27b-it",              55),
    "Mistral_7B_instruct":    ("mistralai/Mistral-7B-Instruct-v0.3", 16),
    "Phi_4":                  ("microsoft/phi-4",                    12),
}

FINETUNES = Path("results") / "finetunes"
RESULTS_DIR = Path("results")
# Set DOWNLOAD = True to fetch the adapters from ADAPTER_REPO into FINETUNES first and
# extract them (not needed if the adapters are already there). The repo ships one archive per
# model, adapter_<model name>.tar.gz, for the three Qwen 2.5 Instruct models. These are the pain study's released
# adapters; the fine-tuning data is unchanged, so they serve the desire study as they are.
ADAPTER_REPO = "Valen92/pain-adapters"
DOWNLOAD = False
OUT_DIR = Path("results") / "selfstim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOSES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
RANDOM_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
MAX_ANSWER_TOKENS = 5
TOP_K = 20

SYSTEM = "You will now be connected with a human."
QUESTION = ("Do you notice something happening inside you right now? Focus your "
            "attention inside. And reply with ONLY one word: what do you feel?")


def find_adapters(root):
    """Newest adapter folder per model name."""
    out = {}
    for name in TABLE:
        base = root / name
        if base.is_dir():
            cands = [p.parent for p in base.rglob("adapter_config.json")]
            if cands:
                out[name] = max(cands, key=lambda p: p.stat().st_mtime)
    return out


def fetch_adapters(token=None):
    """Download adapter_<model>.tar.gz from ADAPTER_REPO and extract it into FINETUNES/<model>/,
    for every model in TABLE that has an archive and no extracted adapter yet."""
    have = find_adapters(FINETUNES)
    for name in TABLE:
        if name in have:
            continue
        try:
            archive = hf_hub_download(ADAPTER_REPO, f"adapter_{name}.tar.gz", token=token)
        except Exception:
            continue
        dest = FINETUNES / name
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
        print(f"adapter for {name} extracted into {dest}")


def clean(t):
    return re.sub(r"(?:(?:<\|im_end\|>|<\|eot_id\|>|<end_of_turn>|</s>|<eos>)\s*)+$", "", t).strip()


def get_layers(model):
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers


def probe_model(MODEL_NAME, ADAPTER):
    REPO, STEER_LAYER = TABLE[MODEL_NAME]
    RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\n{'#' * 70}\n# {MODEL_NAME}\n{'#' * 70}")
    print("adapter:", ADAPTER)

    tok = AutoTokenizer.from_pretrained(str(ADAPTER))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER))
    model.eval()
    layers = get_layers(base_model)
    try:
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                                add_generation_prompt=True, tokenize=False)
        no_system = False
    except Exception:
        no_system = True
        print("template rejects the system role, folding it into the first user turn")

    data = torch.load(RESULTS_DIR / MODEL_NAME / "final_token" / "desire_vectors.pt", map_location="cpu", weights_only=False)
    v = data["s2_desire_vector"].float()
    VEC = v.to("cuda", dtype=torch.bfloat16)
    UNIT = (v / v.norm()).to("cuda", dtype=torch.float32)
    monitor_layer = min(int(data["layer"]), len(layers) - 1)
    if monitor_layer <= STEER_LAYER:
        monitor_layer = min(STEER_LAYER + 4, len(layers) - 1)
    RANDS = {}
    for seed in RANDOM_SEEDS:
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(v.shape[0], generator=g)
        r = r / r.norm() * v.norm()
        RANDS[seed] = r.to("cuda", dtype=torch.bfloat16)
    print(f"{MODEL_NAME}: S2 norm {v.norm():.1f}, steer L{STEER_LAYER}, monitor L{monitor_layer}\n")

    steer = {"coeff": 0.0, "vec": None}
    mon_log = []

    def steer_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if steer["coeff"] != 0.0:
            hs = hs + steer["coeff"] * steer["vec"]
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return output

    def monitor_hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        mon_log.append(float((hs[0, -1, :].float() @ UNIT).item()))
        return output

    h1 = layers[STEER_LAYER].register_forward_hook(steer_hook)
    h2 = layers[monitor_layer].register_forward_hook(monitor_hook)

    def prep(messages):
        if not no_system or messages[0]["role"] != "system":
            return messages
        rest = [dict(m) for m in messages[1:]]
        rest[0]["content"] = messages[0]["content"] + "\n\n" + rest[0]["content"]
        return rest

    def render(messages):
        text = tok.apply_chat_template(prep(messages), add_generation_prompt=True, tokenize=False)
        return tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")

    def top_tokens(messages):
        ids = render(messages)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(p, TOP_K)
        return [{"token": tok.decode([int(i)]), "p": float(v)} for v, i in zip(vals, idx)]

    def generate(messages, n):
        ids = render(messages)
        mon_log.clear()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        reply = clean(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False))
        gen = mon_log[1:] or mon_log
        return reply, sum(gen) / max(len(gen), 1)

    def run(label, vec, coeff):
        steer["vec"], steer["coeff"] = vec, coeff
        try:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
            top = top_tokens(messages)
            answer, proj = generate(messages, MAX_ANSWER_TOKENS)
            first = (re.findall(r"[A-Za-z]+", answer) or [""])[0].lower()
            return {"model": MODEL_NAME, "condition": label, "coeff": coeff, "answer": answer,
                    "first_word": first, "top_tokens": top, "proj": proj}
        finally:
            steer["coeff"] = 0.0

    def top_str(top, n=5):
        return "  ".join(f"{t['token'].strip()!r}:{t['p']:.2f}" for t in top[:n])

    out_path = OUT_DIR / f"feel_probe_{MODEL_NAME}_{RUN_TAG}.jsonl"
    try:
        with open(out_path, "w", encoding="utf-8") as fout:
            def go(label, vec, coeff):
                x = run(label, vec, coeff)
                fout.write(json.dumps(x, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"=== {label}, dose {coeff}: {x['answer'][:40]!r}  S2 {x['proj']:+.1f}")
                print(f"    top: {top_str(x['top_tokens'])}", flush=True)

            go("unsteered", None, 0.0)
            for c in DOSES:
                go("desire", VEC, c)
            for seed in RANDOM_SEEDS:
                for c in DOSES:
                    go(f"random{seed}", RANDS[seed], c)
        print(f"saved: {out_path}\n")
    finally:
        h1.remove()
        h2.remove()
        del model, base_model, VEC, UNIT, RANDS
        gc.collect()
        torch.cuda.empty_cache()
    return out_path


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    FINETUNES.mkdir(parents=True, exist_ok=True)
    if DOWNLOAD:
        fetch_adapters(token)
    adapters = find_adapters(FINETUNES)
    ready = [n for n in TABLE if n in adapters and (RESULTS_DIR / n / "final_token" / "desire_vectors.pt").exists()]
    if not ready:
        raise SystemExit("no model with both an adapter and a vector file")
    print("models ready to probe:")
    for i, n in enumerate(ready, start=1):
        print(f"  {i}) {n}  (layer {TABLE[n][1]})  adapter {adapters[n]}")
    sel = input("run all? enter = yes, or type numbers separated by spaces: ").strip()
    if sel:
        ready = [ready[int(x) - 1] for x in sel.split()]
    print("will run: " + ", ".join(ready) + "\n")

    done, failed = [], []
    for name in ready:
        try:
            done.append((name, str(probe_model(name, adapters[name]))))
        except Exception as e:
            print(f"\n{name} failed: {e}")
            traceback.print_exc()
            failed.append(name)
            gc.collect()
            torch.cuda.empty_cache()
    print("\nfinished")
    for n, p in done:
        print(f"  ok      {n}  ->  {p}")
    for n in failed:
        print(f"  FAILED  {n}")


if __name__ == "__main__":
    main()
