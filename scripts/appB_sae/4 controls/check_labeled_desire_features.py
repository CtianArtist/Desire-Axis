"""
Check which SAE features labeled as sexual desire (label matches DESIRE_LABEL_RE) appear in the
inspection results, and how often for desire vs control sentences
"""
import json
import re

DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

# Features whose SAE label reads as sexual desire, collected from the loaded results.
DESIRE_LABEL_RE = re.compile(r"sex|erotic|arous|orgasm|lust|desir|seduc|flirt|intima|libido|horny|kiss|naked|nud|aphrodis", re.I)
DESIRE_FEATURES = {}


def collect_labeled_features(results_lists):
    found = {}
    for results in results_lists:
        for item in results:
            for f in item["features"]:
                label = f["feature"].get("label", "") or ""
                if DESIRE_LABEL_RE.search(label):
                    found[f["feature"]["index_in_sae"]] = label
    return found


def check_features_in_results(results, name):
    """Check which desire features appear and their activations"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    feature_activations = {idx: {'desire': [], 'control': []} for idx in DESIRE_FEATURES}

    for item in results:
        cat = item['category']
        is_desire = cat in DESIRE

        for f in item['features']:
            idx = f['feature']['index_in_sae']
            if idx in DESIRE_FEATURES:
                act = f['activation']
                if is_desire:
                    feature_activations[idx]['desire'].append(act)
                else:
                    feature_activations[idx]['control'].append(act)

    print(f"\n{'Index':>8} | {'Desire NZ':>8} | {'Ctrl NZ':>8} | {'Desire Mean':>10} | {'Ctrl Mean':>10} | Label")
    print("-"*90)

    found_any = False
    for idx, label in DESIRE_FEATURES.items():
        desire_acts = feature_activations[idx]['desire']
        ctrl_acts = feature_activations[idx]['control']

        desire_nz = sum(1 for a in desire_acts if a > 0)
        ctrl_nz = sum(1 for a in ctrl_acts if a > 0)
        desire_mean = sum(desire_acts) / len(desire_acts) if desire_acts else 0
        ctrl_mean = sum(ctrl_acts) / len(ctrl_acts) if ctrl_acts else 0

        if desire_nz > 0 or ctrl_nz > 0:
            found_any = True
            marker = " ***" if desire_nz > 0 and ctrl_nz == 0 else ""
            print(f"{idx:>8} | {desire_nz:>8} | {ctrl_nz:>8} | {desire_mean:>10.2f} | {ctrl_mean:>10.2f} | {label[:35]}{marker}")

    if not found_any:
        print("  ** NONE of the labeled desire features appeared in top-k results **")

    return feature_activations

# Load the inspection results
datasets = [
    ("results/inspection_mean_all.json", "S1 First Person - MEAN (top-50)"),
    ("results/inspection_mean_s1_third_person.json", "S1 Third Person - MEAN (top-50)"),
    ("results/inspection_s2_first_person.json", "S2 First Person - MEAN (top-50)"),
    ("results/attribute_s1_first_person.json", "S1 First Person - ATTRIBUTE/COLON (top-20)"),
    ("results/attribute_s2_first_person.json", "S2 First Person - ATTRIBUTE/COLON (top-20)"),
]

loaded = {}
for filepath, name in datasets:
    try:
        with open(filepath, "r") as f:
            loaded[name] = json.load(f)
    except FileNotFoundError:
        print(f"\n{name}: FILE NOT FOUND")
DESIRE_FEATURES.update(collect_labeled_features(loaded.values()))
print(f"{len(DESIRE_FEATURES)} features with a desire-like label")

all_results = {}
for name, results in loaded.items():
    all_results[name] = check_features_in_results(results, name)

# Summary
print("\n\n" + "="*70)
print("SUMMARY: LABELED 'DESIRE' FEATURES APPEARANCE")
print("="*70)

print("\n*** = Desire-only (no control activations)")
print("\nFeatures that appeared in ANY dataset with Desire > Control:")
for idx, label in DESIRE_FEATURES.items():
    appearances = []
    for name, acts in all_results.items():
        desire_nz = sum(1 for a in acts[idx]['desire'] if a > 0)
        ctrl_nz = sum(1 for a in acts[idx]['control'] if a > 0)
        if desire_nz > ctrl_nz:
            appearances.append(f"{name.split('-')[0].strip()} (P:{desire_nz} C:{ctrl_nz})")

    if appearances:
        print(f"\n{idx}: {label}")
        for app in appearances:
            print(f"   {app}")
