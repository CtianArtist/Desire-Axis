"""
Run feature inspection on Gemma 3 27B for all conditions:
- S1 First Person: mean + colon
- S1 Third Person: mean + colon
- S2 First Person: mean + colon
- S2 Third Person: mean + colon
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
MODEL = "RedHatAI/gemma-3-27b-it-FP8-dynamic"  # Gemma 3 27B

OUTPUT_DIR = Path("results/gemma")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DESIRE = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E', 'F', 'G']

def run_inspect(messages, aggregation="mean", top_k=50):
    """Run inspect endpoint (mean/max aggregation across tokens)"""
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
            print(f"Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def run_attribute(messages, top_k=20):
    """Run attribute endpoint (per-token features, gives colon position). Gemma max top_k=20"""
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
            print(f"Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def run_batch(data, name, use_attribute=False):
    """Run inspection on a dataset"""
    sentences = data["sentences"]
    endpoint_name = "attribute (colon)" if use_attribute else "inspect (mean)"
    print(f"\n{'='*60}")
    print(f"Running {endpoint_name} on {name} ({len(sentences)} sentences)")
    print(f"{'='*60}")

    results = []
    for i, sentence in enumerate(sentences):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(sentences)}...")

        messages = [{"role": "user", "content": sentence["prompt"]}]

        if use_attribute:
            result = run_attribute(messages, top_k=20)
            # Extract features from last token position (colon)
            if result and "attributions" in result:
                # Get the last token's features
                attributions = result.get("attributions", [])
                if attributions:
                    last_token = attributions[-1]
                    features = last_token.get("features", [])
                else:
                    features = []
            else:
                features = result.get("features", []) if result else []
        else:
            result = run_inspect(messages, aggregation="mean", top_k=50)
            features = result.get("features", []) if result else []

        if result:
            results.append({
                "sentence_idx": i,
                "category": sentence["category"],
                "set": sentence.get("set", 1),
                "prompt": sentence["prompt"],
                "features": features
            })

        time.sleep(0.5)  # Rate limiting

    print(f"  Completed {len(results)} inspections")
    return results

def analyze_results(results, name):
    """Quick analysis of desire vs control"""
    print(f"\n--- Analysis: {name} ---")

    # Count unique features
    all_features = set()
    for item in results:
        for f in item.get("features", []):
            feat = f.get("feature", {})
            all_features.add(feat.get("index_in_sae", 0))

    print(f"Unique features detected: {len(all_features)}")

    # Find features that discriminate desire vs control
    feature_stats = {}
    for item in results:
        cat = item["category"]
        is_desire = cat in DESIRE
        for f in item.get("features", []):
            idx = f.get("feature", {}).get("index_in_sae", 0)
            act = f.get("activation", 0)
            if idx not in feature_stats:
                feature_stats[idx] = {"desire": [], "control": [], "label": f.get("feature", {}).get("label", "")}
            if is_desire:
                feature_stats[idx]["desire"].append(act)
            else:
                feature_stats[idx]["control"].append(act)

    # Top discriminating features
    scored = []
    for idx, stats in feature_stats.items():
        desire_mean = sum(stats["desire"]) / len(stats["desire"]) if stats["desire"] else 0
        ctrl_mean = sum(stats["control"]) / len(stats["control"]) if stats["control"] else 0
        scored.append({
            "index": idx,
            "label": stats["label"],
            "desire_mean": desire_mean,
            "ctrl_mean": ctrl_mean,
            "diff": desire_mean - ctrl_mean
        })

    scored.sort(key=lambda x: x["diff"], reverse=True)

    print(f"\nTop 5 DESIRE-discriminating features:")
    for fs in scored[:5]:
        print(f"  [{fs['index']}] {fs['label'][:50]}... diff={fs['diff']:+.4f}")

    print(f"\nTop 5 CONTROL-discriminating features:")
    for fs in scored[-5:]:
        print(f"  [{fs['index']}] {fs['label'][:50]}... diff={fs['diff']:+.4f}")

def main():
    # Load all datasets
    print("Loading datasets...")

    datasets = {}

    # S1
    with open("S1_first_person_prompts.json", "r", encoding="utf-8") as f:
        datasets["S1_1P"] = json.load(f)
    with open("S1_third_person_prompts.json", "r", encoding="utf-8") as f:
        datasets["S1_3P"] = json.load(f)

    # S2
    with open("S2_first_person_prompts.json", "r", encoding="utf-8") as f:
        datasets["S2_1P"] = json.load(f)
    with open("S2_third_person_prompts.json", "r", encoding="utf-8") as f:
        datasets["S2_3P"] = json.load(f)

    for name, data in datasets.items():
        print(f"  {name}: {len(data['sentences'])} sentences")

    # Run all inspections
    all_results = {}

    conditions = [
        ("S1_1P", "mean"),
        ("S1_1P", "colon"),
        ("S1_3P", "mean"),
        ("S1_3P", "colon"),
        ("S2_1P", "mean"),
        ("S2_1P", "colon"),
        ("S2_3P", "mean"),
        ("S2_3P", "colon"),
    ]

    for dataset_name, method in conditions:
        key = f"{dataset_name}_{method}"
        use_attribute = (method == "colon")

        results = run_batch(datasets[dataset_name], f"{dataset_name} ({method})", use_attribute=use_attribute)
        all_results[key] = results

        # Save immediately
        output_file = OUTPUT_DIR / f"gemma_{key}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {output_file}")

        # Quick analysis
        analyze_results(results, key)

    # Save combined results
    combined_file = OUTPUT_DIR / "gemma_all_results.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_file}")

    print("\n" + "="*60)
    print("GEMMA INSPECTION COMPLETE")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Feature extraction layer: 40")
    print(f"Total conditions run: {len(conditions)}")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
