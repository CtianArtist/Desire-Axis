"""Writes the prompt files the SAE scripts read, from datasets/3.1_desire_and_control_datasets.json:
S1_first_person_prompts.json, S1_third_person_prompts.json, S2_first_person_prompts.json and
S2_third_person_prompts.json, each {"metadata": ..., "sentences": [...]}, plus an empty results/.

Run it from the folder of the SAE script you are about to run (the scripts open these files
from the current directory), e.g.:
  cd "scripts/appB_sae/1 contrasts and inspection Llama 3.3 70B L50" && python ../00_export_prompts.py
"""

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "3.1_desire_and_control_datasets.json"
FILES = {"S1_1P": "S1_first_person_prompts.json", "S1_3P": "S1_third_person_prompts.json",
         "S2_1P": "S2_first_person_prompts.json", "S2_3P": "S2_third_person_prompts.json"}

data = json.load(open(DATASET, encoding="utf-8"))
meta = {k: data["metadata"][k] for k in ("description", "categories", "category_labels")}
out_dir = Path.cwd()
for key, name in FILES.items():
    sentences = data["datasets"][key]["sentences"]
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump({"metadata": {**meta, "set": key}, "sentences": sentences}, f, indent=2, ensure_ascii=False)
    print(f"wrote {name}: {len(sentences)} sentences")
(out_dir / "results").mkdir(exist_ok=True)
