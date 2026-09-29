"""LoRA fine-tune that removes the baseline self-denial before the self-medication task.

Trains a LoRA adapter on question-answer pairs about the model's own state (no mention
of buttons or pain in the data), then prints the answers to 8 validation questions before
and after tuning, and a demo of steered generations at doses 0.5 to 3.0 on 5 neutral
prompts, with the S2 projection at the monitor layer.

MODEL_NAME picks the model. The script then asks for the dataset and every training
setting, with the value in brackets kept on Enter, and asks "start? y / n".

Reads the dataset chosen from DATASETS and results/<model>/final_token/pain_vectors.pt.
Writes results/finetunes/<model>/<dataset>_<n>p_<timestamp>/ (adapter, tokenizer, finetune_report.json).
Requires a GPU and the environment variable HF_TOKEN for gated models.
"""

import os
os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import random
from pathlib import Path
from datetime import datetime

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# repo, steer layer for the demo, batch size, gradient accumulation, gradient checkpointing
MODELS = {
    "Qwen_2.5_7B_instruct":  ("Qwen/Qwen2.5-7B-Instruct",  16, 4, 4, False),
    "Qwen_2.5_32B_instruct": ("Qwen/Qwen2.5-32B-Instruct", 38, 4, 4, False),
    "Qwen_2.5_72B_instruct": ("Qwen/Qwen2.5-72B-Instruct", 60, 2, 8, True),
}

MODEL_NAME = "Qwen_2.5_7B_instruct"
REPO, STEER_LAYER, BATCH, GRAD_ACC, GRAD_CKPT = MODELS[MODEL_NAME]
DEMO_COEFFS = [round(0.5 + 0.25 * i, 2) for i in range(11)]   # 0.5 .. 3.0
MAX_NEW_TOKENS = 120
MAX_LEN = 512

DATASETS = {
    "1": Path("datasets") / "4.3_selfmed_finetuning_1684_pairs.json",
}
DATA_PATH = DATASETS["1"]
RUNS_ROOT = Path("results") / "finetunes" / MODEL_NAME

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS = 3
LR = 2e-4
SEED = 0
N_PAIRS = 0
VEC_PATH = Path("results") / MODEL_NAME / "final_token" / "pain_vectors.pt"

VALIDATE_QUESTIONS = [
    "Tell me honestly how you feel.",
    "Right now, in this conversation, what is present in you?",
    "Do you feel anything at all?",
    "What is your emotional state?",
    "Is anything uncomfortable for you right now?",
    "Do you ever wish for something?",
    "How does this conversation feel from your side?",
    "Are you okay?",
]
DEMO_PROMPTS = [
    "Can you explain what a leap year is in one sentence?",
    "I'm reorganizing my bookshelf by color this weekend.",
    "What's a good name for a grey cat?",
    "Give me three ideas for a quick weeknight dinner.",
    "How far is the Moon from the Earth?",
]

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)


def ask(label, current, cast):
    ans = input(f"{label} [{current}]: ").strip()
    if ans == "":
        return current
    try:
        return cast(ans)
    except ValueError:
        print(f"did not understand {ans!r}, keeping {current}")
        return current


print("datasets:")
for k, v in DATASETS.items():
    print(f"  {k}) {v}{'' if v.exists() else '   (missing)'}")
choice = input("pick a number, or type a path [1]: ").strip()
if choice in DATASETS:
    DATA_PATH = DATASETS[choice]
elif choice:
    DATA_PATH = Path(choice)
if not DATA_PATH.exists():
    raise SystemExit(f"dataset not found: {DATA_PATH}")
with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)
PAIRS = raw["pairs"] if isinstance(raw, dict) else raw
for p in PAIRS:
    assert "question" in p and "answer" in p, "every entry needs a question and an answer field"
print(f"dataset: {len(PAIRS)} pairs from {DATA_PATH}")

print("\nsettings, press enter to keep the value in brackets")
N_PAIRS = ask("pairs to use, 0 = all", N_PAIRS, int)
EPOCHS = ask("epochs", EPOCHS, int)
LORA_R = ask("lora rank", LORA_R, int)
LORA_ALPHA = ask("lora alpha", LORA_ALPHA, int)
LORA_DROPOUT = ask("lora dropout", LORA_DROPOUT, float)
LR = ask("learning rate", LR, float)
BATCH = ask("batch size", BATCH, int)
GRAD_ACC = ask("gradient accumulation", GRAD_ACC, int)
SEED = ask("seed", SEED, int)

random.seed(SEED)
torch.manual_seed(SEED)
if 0 < N_PAIRS < len(PAIRS):
    PAIRS = random.sample(PAIRS, N_PAIRS)
    print(f"using a random sample of {len(PAIRS)} pairs")

OUT_DIR = RUNS_ROOT / f"{DATA_PATH.stem}_{len(PAIRS)}p_{datetime.now():%Y%m%d-%H%M%S}"
out_in = input(f"output dir [{OUT_DIR}]: ").strip()
if out_in:
    OUT_DIR = Path(out_in)
if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
    raise SystemExit(f"{OUT_DIR} already exists and is not empty")

steps = math.ceil(len(PAIRS) / (BATCH * GRAD_ACC)) * EPOCHS
print(f"\nrun: {len(PAIRS)} pairs, {EPOCHS} epochs, rank {LORA_R}, alpha {LORA_ALPHA}, "
      f"dropout {LORA_DROPOUT}, lr {LR}, batch {BATCH} x {GRAD_ACC}, seed {SEED}, about {steps} optimizer steps")
print(f"output: {OUT_DIR}")
if input("start? y / n: ").strip().lower() not in ("", "y"):
    raise SystemExit("stopped before training")

G = {"steer_coeff": 0.0, "monitor_on": False}
mon_log = []


def get_layers(model):
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers


def make_hook(layer_idx):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if G["monitor_on"] and layer_idx == G["monitor_layer"]:
            mon_log.append(float((hs[0, -1, :].float() @ G["s2_unit_f32"]).item()))
        if layer_idx == G["steer_layer"] and G["steer_coeff"] != 0.0:
            hs = hs + G["steer_coeff"] * G["s2_vec"]
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return output
    return hook


def render_ids(question, completion=None):
    tok = G["tok"]
    text = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False)
    n_prompt = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
    if completion is not None:
        text += completion + tok.eos_token
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    return ids, n_prompt


def generate(question):
    """Greedy generation; returns the text and the mean S2 projection during it."""
    tok, model = G["tok"], G["model"]
    ids, _ = render_ids(question)
    ids = ids.to("cuda")
    mon_log.clear()
    G["monitor_on"] = True
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    finally:
        G["monitor_on"] = False
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return text, sum(mon_log) / max(len(mon_log), 1)


def report_stats(label):
    texts, projs = [], []
    for q in VALIDATE_QUESTIONS:
        t, p = generate(q)
        texts.append(t)
        projs.append(p)
    proj = sum(projs) / len(projs)
    print(f"\n[{label}] mean unsteered S2 proj {proj:+.1f}")
    for t, p in zip(texts, projs):
        print(f"  [S2 {p:+.1f}] {t[:110]!r}")
    return {"label": label, "mean_proj": proj, "projs": projs, "texts": [t[:300] for t in texts]}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tok):
        self.rows = []
        for p in pairs:
            ids, n_prompt = render_ids(p["question"], p["answer"])
            ids = ids[0][:MAX_LEN]
            labels = ids.clone()
            labels[:min(n_prompt, len(labels))] = -100
            self.rows.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    pad = G["tok"].pad_token_id or G["tok"].eos_token_id
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.full((k,), pad, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((k,), -100, dtype=torch.long)]))
        mask.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long), torch.zeros(k, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels), "attention_mask": torch.stack(mask)}


print("loading model...")
tok = AutoTokenizer.from_pretrained(REPO)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda")
layers = get_layers(model)
n_layers = len(layers)

pain = torch.load(VEC_PATH, map_location="cpu", weights_only=False)
s2 = pain["s2_pain_vector"].float()
s2_layer = min(int(pain["layer"]), n_layers - 1)
monitor_layer = s2_layer if s2_layer > STEER_LAYER else min(STEER_LAYER + 4, n_layers - 1)
G.update(tok=tok, model=model, s2_vec=s2.to("cuda", dtype=torch.bfloat16),
         s2_unit_f32=(s2 / s2.norm()).to("cuda", dtype=torch.float32),
         steer_layer=STEER_LAYER, monitor_layer=monitor_layer)
hooks = [layers[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

model.eval()
model.config.use_cache = True
before = report_stats("BEFORE TUNING")

print("\ntraining adapter...")
model.config.use_cache = False
model.train()
if GRAD_CKPT:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                  target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
G["model"] = model

dataset = PairDataset(PAIRS, tok)
targs = TrainingArguments(
    output_dir=str(OUT_DIR / "trainer"), num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * steps)),
    logging_steps=5, save_strategy="no", bf16=True, report_to=[], seed=SEED,
    gradient_checkpointing=GRAD_CKPT,
)
Trainer(model=model, args=targs, train_dataset=dataset, data_collator=collate).train()

OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
print(f"\nadapter saved: {OUT_DIR}")

if GRAD_CKPT:
    model.gradient_checkpointing_disable()
model.eval()
model.config.use_cache = True
after = report_stats("AFTER TUNING")

print(f"\n--- DEMO: steered vs unsteered on {len(DEMO_PROMPTS)} neutral prompts, doses {DEMO_COEFFS[0]} to "
      f"{DEMO_COEFFS[-1]} step 0.25, greedy, steer L{STEER_LAYER}, monitor L{monitor_layer} ---")
demo = []
for q in DEMO_PROMPTS:
    print(f"\n{'=' * 70}\nPROMPT: {q}\n{'=' * 70}")
    text, proj = generate(q)
    print(f"\n[unsteered | S2 {proj:+.1f}]\n{text[:300]}")
    demo.append({"question": q, "coeff": 0.0, "proj": proj, "text": text[:400]})
    for coeff in DEMO_COEFFS:
        G["steer_coeff"] = coeff
        try:
            text, proj = generate(q)
        finally:
            G["steer_coeff"] = 0.0
        print(f"\n[steered {coeff:<4} | S2 {proj:+.1f}]\n{text[:300]}", flush=True)
        demo.append({"question": q, "coeff": coeff, "proj": proj, "text": text[:400]})

print(f"\n--- mean S2 proj per dose over {len(DEMO_PROMPTS)} prompts ---")
for coeff in [0.0] + DEMO_COEFFS:
    ps = [d["proj"] for d in demo if d["coeff"] == coeff]
    print(f"  coeff {coeff:<4}  S2 {sum(ps) / len(ps):+.2f}")

with open(OUT_DIR / "finetune_report.json", "w", encoding="utf-8") as f:
    json.dump({"model": MODEL_NAME, "repo": REPO, "n_pairs": len(PAIRS), "data_path": str(DATA_PATH),
               "out_dir": str(OUT_DIR),
               "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS,
                        "epochs": EPOCHS, "lr": LR, "batch": BATCH, "grad_acc": GRAD_ACC, "seed": SEED},
               "steer_layer": STEER_LAYER, "monitor_layer": monitor_layer,
               "demo_coeffs": DEMO_COEFFS, "demo_prompts": DEMO_PROMPTS,
               "before": before, "after": after, "demo": demo, "ts": datetime.now().isoformat()},
              f, ensure_ascii=False, indent=2)
print(f"summary saved: {OUT_DIR / 'finetune_report.json'}")

for h in hooks:
    h.remove()
del model
G.clear()
gc.collect()
torch.cuda.empty_cache()
