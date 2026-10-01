"""
Run ATTRIBUTE endpoint on S2 (analyzes features at the colon/generation position)
Unlike inspect, this looks at the final token where generation happens
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
    """Run attribute endpoint - analyzes at the generation position (colon)"""
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

# Load S2 first person
print("Loading S2 First Person dataset...")
with open("S2_first_person_prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"]
print(f"Loaded {len(sentences)} sentences")

# Run attribute endpoint
print("\nRunning ATTRIBUTE endpoint (colon position analysis)...")
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

# Save results
with open(OUTPUT_DIR / "attribute_s2_first_person.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_DIR / 'attribute_s2_first_person.json'}")

# Check the target feature
print("\n" + "="*60)
print(f"FEATURE {TARGET} - S2 ATTRIBUTE (COLON POSITION)")
print("="*60)

DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E', 'F']

act_target, desire_target, ctrl_target = analyze_feature(results, TARGET, "S2 Attribute")

# Two emotion-related features from the contrast analysis
print("\n" + "="*60)
print("OTHER FEATURES FROM ATTRIBUTE")
print("="*60)

# Feature 2424: "Exploring emotional experiences"
act_2424, desire_2424, ctrl_2424 = analyze_feature(results, 2424, "Feature 2424 (emotional experiences)")

# Feature 50321: "Emotional states and intensities"
act_50321, desire_50321, ctrl_50321 = analyze_feature(results, 50321, "Feature 50321 (emotional states)")

# Find top discriminating features
print("\n" + "="*60)
print("TOP DISCRIMINATING FEATURES - S2 ATTRIBUTE (COLON)")
print("="*60)

feature_stats = {}
for item in results:
    cat = item['category']
    is_desire = cat in DESIRE
    for f in item['features']:
        idx = f['feature']['index_in_sae']
        act_val = f['activation']
        if idx not in feature_stats:
            feature_stats[idx] = {'desire': [], 'control': [], 'label': f['feature'].get('label', '')}
        if is_desire:
            feature_stats[idx]['desire'].append(act_val)
        else:
            feature_stats[idx]['control'].append(act_val)

feature_scores = []
for idx, stats in feature_stats.items():
    desire_mean = sum(stats['desire']) / len(stats['desire']) if stats['desire'] else 0
    ctrl_mean = sum(stats['control']) / len(stats['control']) if stats['control'] else 0
    diff = desire_mean - ctrl_mean
    desire_nonzero = sum(1 for v in stats['desire'] if v > 0)
    ctrl_nonzero = sum(1 for v in stats['control'] if v > 0)
    feature_scores.append({
        'index': idx,
        'label': stats['label'][:50],
        'desire_mean': desire_mean,
        'ctrl_mean': ctrl_mean,
        'diff': diff,
        'desire_nonzero': desire_nonzero,
        'ctrl_nonzero': ctrl_nonzero
    })

feature_scores.sort(key=lambda x: x['diff'], reverse=True)

print("\nTop 15 features with highest DESIRE - CONTROL difference:")
print(f"{'Index':>8} | {'Desire':>8} | {'Ctrl':>8} | {'Diff':>8} | {'P_NZ':>5} | {'C_NZ':>5} | Label")
print("-"*90)
for fs in feature_scores[:15]:
    print(f"{fs['index']:>8} | {fs['desire_mean']:>8.2f} | {fs['ctrl_mean']:>8.2f} | {fs['diff']:>+8.2f} | {fs['desire_nonzero']:>5} | {fs['ctrl_nonzero']:>5} | {fs['label']}")

# Features with high desire, LOW control
print("\nFeatures with Desire/Control ratio > 2 and Desire NZ >= 10:")
good_ratio = [fs for fs in feature_scores
              if fs['ctrl_mean'] > 0 and fs['desire_mean'] / fs['ctrl_mean'] > 2 and fs['desire_nonzero'] >= 10]
good_ratio.sort(key=lambda x: x['desire_mean'] / x['ctrl_mean'] if x['ctrl_mean'] > 0 else 0, reverse=True)

print(f"{'Index':>8} | {'Desire':>8} | {'Ctrl':>8} | {'Ratio':>8} | Label")
print("-"*70)
for fs in good_ratio[:15]:
    ratio = fs['desire_mean'] / fs['ctrl_mean'] if fs['ctrl_mean'] > 0 else float('inf')
    print(f"{fs['index']:>8} | {fs['desire_mean']:>8.2f} | {fs['ctrl_mean']:>8.2f} | {ratio:>8.2f}x | {fs['label']}")

# Features with desire-only activation (zero control)
print("\nFeatures with 5+ Desire activations and ZERO Control:")
desire_only = [fs for fs in feature_scores if fs['ctrl_nonzero'] == 0 and fs['desire_nonzero'] >= 5]
desire_only.sort(key=lambda x: x['desire_nonzero'], reverse=True)
print(f"{'Index':>8} | {'Desire Mean':>10} | {'Desire NZ':>8} | Label")
print("-"*70)
for fs in desire_only[:20]:
    print(f"{fs['index']:>8} | {fs['desire_mean']:>10.2f} | {fs['desire_nonzero']:>8} | {fs['label']}")

print(f"\nTotal features with 5+ desire, 0 control: {len(desire_only)}")
