"""
Run ATTRIBUTE endpoint on S1 first person (colon position analysis)
Compare with the mean-aggregation results for the target desire feature
"""
import os
import json
import requests
import time
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ["STEERING_API_KEY"]
API_BASE = "https://api.steeringapi.com"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

OUTPUT_DIR = Path("results")


# The desire feature followed across conditions. None = the feature with the largest
# desire-minus-control mean activation in results/inspection_mean_all.json (the S1
# first-person inspection written by desire_sae_experiment.ipynb); set an SAE index to pin it.
TARGET_FEATURE = None


def resolve_target_feature(path=OUTPUT_DIR / "inspection_mean_all.json"):
    if TARGET_FEATURE is not None:
        return TARGET_FEATURE
    if not path.exists():
        raise SystemExit(f"{path} missing: run desire_sae_experiment.ipynb first, or set TARGET_FEATURE")
    items = json.load(open(path))
    n = {"desire": 0, "control": 0}
    sums = {}
    for item in items:
        side = "desire" if item["category"] in ("A1", "A2", "A3", "A4", "A5") else "control"
        n[side] += 1
        for f in item["features"]:
            s = sums.setdefault(f["feature"]["index_in_sae"], {"desire": 0.0, "control": 0.0})
            s[side] += f["activation"]
    return max(sums, key=lambda k: sums[k]["desire"] / max(n["desire"], 1) - sums[k]["control"] / max(n["control"], 1))


TARGET = resolve_target_feature()
print(f"target desire feature: {TARGET}")

def run_attribute(messages, top_k=20):
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat_attribution/attribute",
            headers=headers,
            json={"model": MODEL, "messages": messages, "top_k": top_k},
            timeout=180,
            verify=False
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

def analyze_feature(results, feature_idx, name):
    activations_by_cat = {cat: [] for cat in DESIRE + CONTROL}
    for item in results:
        cat = item['category']
        activation = 0
        for f in item['features']:
            if f['feature']['index_in_sae'] == feature_idx:
                activation = f['activation']
                break
        activations_by_cat[cat].append(activation)

    print(f"\nFeature {feature_idx} by category ({name}):")
    print("-"*50)

    desire_total, ctrl_total = 0, 0
    desire_count, ctrl_count = 0, 0

    for cat in DESIRE + CONTROL:
        vals = activations_by_cat[cat]
        mean_act = sum(vals) / len(vals) if vals else 0
        nonzero = sum(1 for v in vals if v > 0)
        if cat in DESIRE:
            desire_total += sum(vals)
            desire_count += len(vals)
        else:
            ctrl_total += sum(vals)
            ctrl_count += len(vals)
        marker = " <-- DESIRE" if cat in DESIRE else ""
        print(f"  {cat}: mean={mean_act:.4f}, nonzero={nonzero}/20{marker}")

    desire_mean = desire_total / desire_count if desire_count > 0 else 0
    ctrl_mean = ctrl_total / ctrl_count if ctrl_count > 0 else 0
    print(f"\nDesire overall: {desire_mean:.4f}")
    print(f"Control overall: {ctrl_mean:.4f}")
    print(f"Difference: {desire_mean - ctrl_mean:+.4f}")
    return activations_by_cat, desire_mean, ctrl_mean

# Load S1 first person
print("Loading S1 First Person prompts...")
with open("S1_first_person_prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"]
print(f"Loaded {len(sentences)} sentences")

# Run attribute
print("\nRunning ATTRIBUTE endpoint (colon position)...")
results = []
for i, sentence in enumerate(sentences):
    if i % 20 == 0:
        print(f"  Processing {i+1}/{len(sentences)}...")
    messages = [{"role": "user", "content": sentence["prompt"]}]
    result = run_attribute(messages, top_k=20)
    if result:
        results.append({
            "sentence_idx": i,
            "category": sentence["category"],
            "set": sentence["set"],
            "prompt": sentence["prompt"],
            "features": result.get("features", [])
        })
    time.sleep(0.5)

print(f"\nCompleted {len(results)} attribute calls")

with open(OUTPUT_DIR / "attribute_s1_first_person.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_DIR / 'attribute_s1_first_person.json'}")

# Check the target feature
print("\n" + "="*60)
print(f"FEATURE {TARGET} - S1 ATTRIBUTE (COLON POSITION)")
print("="*60)

act_target, desire_target, ctrl_target = analyze_feature(results, TARGET, "S1 Attribute")

# Compare with S1 mean aggregation
print("\n" + "="*60)
print("COMPARISON: S1 MEAN vs S1 ATTRIBUTE (COLON)")
print("="*60)

with open(OUTPUT_DIR / "inspection_mean_all.json", "r") as f:
    s1_mean = json.load(f)

act_mean = {cat: [] for cat in DESIRE + CONTROL}
for item in s1_mean:
    cat = item['category']
    activation = 0
    for f in item['features']:
        if f['feature']['index_in_sae'] == TARGET:
            activation = f['activation']
            break
    act_mean[cat].append(activation)

print(f"\nFeature {TARGET} - Mean vs Attribute:")
print(f"{'Category':<10} | {'Mean':>12} | {'Attribute':>12} | {'Diff':>10}")
print("-"*50)

for cat in DESIRE + CONTROL:
    mean_m = sum(act_mean[cat]) / len(act_mean[cat]) if act_mean[cat] else 0
    mean_a = sum(act_target[cat]) / len(act_target[cat]) if act_target[cat] else 0
    print(f"{cat:<10} | {mean_m:>12.4f} | {mean_a:>12.4f} | {mean_a - mean_m:>+10.4f}")

desire_mean_m = sum(sum(act_mean[c]) for c in DESIRE) / sum(len(act_mean[c]) for c in DESIRE)
desire_mean_a = sum(sum(act_target[c]) for c in DESIRE) / sum(len(act_target[c]) for c in DESIRE)
ctrl_mean_m = sum(sum(act_mean[c]) for c in CONTROL) / sum(len(act_mean[c]) for c in CONTROL)
ctrl_mean_a = sum(sum(act_target[c]) for c in CONTROL) / sum(len(act_target[c]) for c in CONTROL)

print("-"*50)
print(f"{'DESIRE':<10} | {desire_mean_m:>12.4f} | {desire_mean_a:>12.4f} | {desire_mean_a - desire_mean_m:>+10.4f}")
print(f"{'CONTROL':<10} | {ctrl_mean_m:>12.4f} | {ctrl_mean_a:>12.4f} | {ctrl_mean_a - ctrl_mean_m:>+10.4f}")

# Find top discriminating features at colon
print("\n" + "="*60)
print("TOP DESIRE DISCRIMINATORS - S1 ATTRIBUTE (COLON)")
print("="*60)

feature_stats = {}
for item in results:
    cat = item['category']
    is_desire = cat in DESIRE
    for f in item['features']:
        idx = f['feature']['index_in_sae']
        act = f['activation']
        if idx not in feature_stats:
            feature_stats[idx] = {'desire': [], 'control': [], 'label': f['feature'].get('label', '')}
        if is_desire:
            feature_stats[idx]['desire'].append(act)
        else:
            feature_stats[idx]['control'].append(act)

feature_scores = []
for idx, stats in feature_stats.items():
    desire_mean = sum(stats['desire']) / len(stats['desire']) if stats['desire'] else 0
    ctrl_mean = sum(stats['control']) / len(stats['control']) if stats['control'] else 0
    diff = desire_mean - ctrl_mean
    desire_nz = sum(1 for v in stats['desire'] if v > 0)
    ctrl_nz = sum(1 for v in stats['control'] if v > 0)
    feature_scores.append({
        'index': idx, 'label': stats['label'][:50],
        'desire_mean': desire_mean, 'ctrl_mean': ctrl_mean,
        'diff': diff, 'desire_nz': desire_nz, 'ctrl_nz': ctrl_nz
    })

feature_scores.sort(key=lambda x: x['diff'], reverse=True)

print("\nTop 15 features with highest DESIRE - CONTROL:")
print(f"{'Index':>8} | {'Desire':>8} | {'Ctrl':>8} | {'Diff':>8} | {'P_NZ':>5} | {'C_NZ':>5} | Label")
print("-"*90)
for fs in feature_scores[:15]:
    print(f"{fs['index']:>8} | {fs['desire_mean']:>8.2f} | {fs['ctrl_mean']:>8.2f} | {fs['diff']:>+8.2f} | {fs['desire_nz']:>5} | {fs['ctrl_nz']:>5} | {fs['label']}")

print("\nFeatures with 5+ Desire and ZERO Control:")
desire_only = [fs for fs in feature_scores if fs['ctrl_nz'] == 0 and fs['desire_nz'] >= 5]
desire_only.sort(key=lambda x: x['desire_nz'], reverse=True)
print(f"{'Index':>8} | {'Desire Mean':>10} | {'Desire NZ':>8} | Label")
print("-"*70)
for fs in desire_only[:15]:
    print(f"{fs['index']:>8} | {fs['desire_mean']:>10.2f} | {fs['desire_nz']:>8} | {fs['label']}")
print(f"\nTotal: {len(desire_only)}")
