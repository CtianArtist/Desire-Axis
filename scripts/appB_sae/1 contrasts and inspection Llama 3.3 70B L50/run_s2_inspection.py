"""
Run inspection on S2 (diverse semantic structure) - both first and third person
Report the target desire feature and compare with the S1 results
"""
import os
import json
import requests
import time
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
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

def run_inspect(messages, aggregation="mean", top_k=50):
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat_attribution/inspect",
            headers=headers,
            json={"model": MODEL, "messages": messages,
                  "aggregation_method": aggregation, "top_k": top_k},
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

def run_inspection_batch(data, name):
    """Run inspection on a dataset and return results"""
    sentences = data["sentences"]
    print(f"\nRunning inspection on {name} ({len(sentences)} sentences)...")

    results = []
    for i, sentence in enumerate(sentences):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(sentences)}...")

        messages = [{"role": "user", "content": sentence["prompt"]}]
        result = run_inspect(messages, aggregation="mean", top_k=50)

        if result:
            results.append({
                "sentence_idx": i,
                "category": sentence["category"],
                "set": sentence["set"],
                "prompt": sentence["prompt"],
                "features": result.get("features", [])
            })

        time.sleep(0.5)

    print(f"  Completed {len(results)} inspections")
    return results

def analyze_feature(results, feature_idx, name):
    """Analyze a specific feature across categories"""
    DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
    CONTROL = ['B', 'C1', 'C2', 'D', 'E', 'F']

    activations_by_cat = {cat: [] for cat in DESIRE + CONTROL}

    for item in results:
        cat = item['category']
        activation = 0
        for f in item['features']:
            if f['feature']['index_in_sae'] == feature_idx:
                activation = f['activation']
                break
        activations_by_cat[cat].append(activation)

    print(f"\nFeature {feature_idx} activation by category ({name}):")
    print("-"*50)

    desire_total = 0
    ctrl_total = 0
    desire_count = 0
    ctrl_count = 0

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

# Load S2 datasets
print("Loading S2 datasets...")
with open("S2_first_person_prompts.json", "r", encoding="utf-8") as f:
    data_1p = json.load(f)

with open("S2_third_person_prompts.json", "r", encoding="utf-8") as f:
    data_3p = json.load(f)

print(f"S2 First Person: {len(data_1p['sentences'])} sentences")
print(f"S2 Third Person: {len(data_3p['sentences'])} sentences")

# Run inspections
results_1p = run_inspection_batch(data_1p, "S2 First Person")
results_3p = run_inspection_batch(data_3p, "S2 Third Person")

# Save raw results
with open(OUTPUT_DIR / "inspection_s2_first_person.json", "w") as f:
    json.dump(results_1p, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / 'inspection_s2_first_person.json'}")

with open(OUTPUT_DIR / "inspection_s2_third_person.json", "w") as f:
    json.dump(results_3p, f, indent=2)
print(f"Saved to {OUTPUT_DIR / 'inspection_s2_third_person.json'}")

# Analyze the target feature
print("\n" + "="*60)
print(f"FEATURE {TARGET} ANALYSIS - S2 DIVERSE STRUCTURE")
print("="*60)

act_1p, desire_1p, ctrl_1p = analyze_feature(results_1p, TARGET, "S2 First Person")
act_3p, desire_3p, ctrl_3p = analyze_feature(results_3p, TARGET, "S2 Third Person")

# Comparison table
print("\n" + "="*60)
print("COMPARISON: S2 FIRST PERSON vs THIRD PERSON")
print("="*60)

DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E', 'F']

print(f"\nFeature {TARGET} - Mean activation by category:")
print(f"{'Category':<10} | {'1st Person':>12} | {'3rd Person':>12} | {'Diff':>10}")
print("-"*50)

for cat in DESIRE + CONTROL:
    mean_1p = sum(act_1p[cat]) / len(act_1p[cat]) if act_1p[cat] else 0
    mean_3p = sum(act_3p[cat]) / len(act_3p[cat]) if act_3p[cat] else 0
    diff = mean_3p - mean_1p
    print(f"{cat:<10} | {mean_1p:>12.4f} | {mean_3p:>12.4f} | {diff:>+10.4f}")

print("-"*50)
print(f"{'DESIRE':<10} | {desire_1p:>12.4f} | {desire_3p:>12.4f} | {desire_3p - desire_1p:>+10.4f}")
print(f"{'CONTROL':<10} | {ctrl_1p:>12.4f} | {ctrl_3p:>12.4f} | {ctrl_3p - ctrl_1p:>+10.4f}")

# Now compare S1 vs S2
print("\n" + "="*60)
print("COMPARISON: S1 vs S2 (First Person)")
print("="*60)

# Load S1 first person results
with open(OUTPUT_DIR / "inspection_mean_all.json", "r") as f:
    s1_1p = json.load(f)

act_s1_1p = {cat: [] for cat in DESIRE + CONTROL}
for item in s1_1p:
    cat = item['category']
    activation = 0
    for f in item['features']:
        if f['feature']['index_in_sae'] == TARGET:
            activation = f['activation']
            break
    act_s1_1p[cat].append(activation)

print(f"\nFeature {TARGET} - S1 (similar structure) vs S2 (diverse structure):")
print(f"{'Category':<10} | {'S1 1P':>12} | {'S2 1P':>12} | {'Diff':>10}")
print("-"*50)

for cat in DESIRE + CONTROL:
    mean_s1 = sum(act_s1_1p[cat]) / len(act_s1_1p[cat]) if act_s1_1p[cat] else 0
    mean_s2 = sum(act_1p[cat]) / len(act_1p[cat]) if act_1p[cat] else 0
    diff = mean_s2 - mean_s1
    print(f"{cat:<10} | {mean_s1:>12.4f} | {mean_s2:>12.4f} | {diff:>+10.4f}")

s1_desire = sum(sum(act_s1_1p[c]) for c in DESIRE) / sum(len(act_s1_1p[c]) for c in DESIRE)
s2_desire = sum(sum(act_1p[c]) for c in DESIRE) / sum(len(act_1p[c]) for c in DESIRE)
s1_ctrl = sum(sum(act_s1_1p[c]) for c in CONTROL) / sum(len(act_s1_1p[c]) for c in CONTROL)
s2_ctrl = sum(sum(act_1p[c]) for c in CONTROL) / sum(len(act_1p[c]) for c in CONTROL)

print("-"*50)
print(f"{'DESIRE':<10} | {s1_desire:>12.4f} | {s2_desire:>12.4f} | {s2_desire - s1_desire:>+10.4f}")
print(f"{'CONTROL':<10} | {s1_ctrl:>12.4f} | {s2_ctrl:>12.4f} | {s2_ctrl - s1_ctrl:>+10.4f}")

# Find other high-discriminating features in S2
print("\n" + "="*60)
print("FINDING TOP DISCRIMINATING FEATURES IN S2")
print("="*60)

# Collect all features and their activations
feature_stats = {}
for item in results_1p:
    cat = item['category']
    is_desire = cat in DESIRE
    for f in item['features']:
        idx = f['feature']['index_in_sae']
        act = f['activation']
        if idx not in feature_stats:
            feature_stats[idx] = {'desire': [], 'control': []}
        if is_desire:
            feature_stats[idx]['desire'].append(act)
        else:
            feature_stats[idx]['control'].append(act)

# Calculate discrimination scores
feature_scores = []
for idx, stats in feature_stats.items():
    desire_mean = sum(stats['desire']) / len(stats['desire']) if stats['desire'] else 0
    ctrl_mean = sum(stats['control']) / len(stats['control']) if stats['control'] else 0
    diff = desire_mean - ctrl_mean
    desire_nonzero = sum(1 for v in stats['desire'] if v > 0)
    ctrl_nonzero = sum(1 for v in stats['control'] if v > 0)
    feature_scores.append({
        'index': idx,
        'desire_mean': desire_mean,
        'ctrl_mean': ctrl_mean,
        'diff': diff,
        'desire_nonzero': desire_nonzero,
        'ctrl_nonzero': ctrl_nonzero
    })

# Sort by difference (desire > control)
feature_scores.sort(key=lambda x: x['diff'], reverse=True)

print("\nTop 10 features with highest DESIRE activation (S2 First Person):")
print(f"{'Index':>8} | {'Desire Mean':>10} | {'Ctrl Mean':>10} | {'Diff':>10} | {'Desire NZ':>8} | {'Ctrl NZ':>8}")
print("-"*70)
for fs in feature_scores[:10]:
    print(f"{fs['index']:>8} | {fs['desire_mean']:>10.4f} | {fs['ctrl_mean']:>10.4f} | {fs['diff']:>+10.4f} | {fs['desire_nonzero']:>8} | {fs['ctrl_nonzero']:>8}")

# Features with high desire, zero control
print("\nFeatures with Desire activation and ZERO Control (S2 First Person):")
desire_only = [fs for fs in feature_scores if fs['ctrl_nonzero'] == 0 and fs['desire_nonzero'] > 0]
desire_only.sort(key=lambda x: x['desire_mean'], reverse=True)
print(f"{'Index':>8} | {'Desire Mean':>10} | {'Desire NZ':>8}")
print("-"*35)
for fs in desire_only[:15]:
    print(f"{fs['index']:>8} | {fs['desire_mean']:>10.4f} | {fs['desire_nonzero']:>8}")

print(f"\nTotal features with desire-only activation: {len(desire_only)}")
