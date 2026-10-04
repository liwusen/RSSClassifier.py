import json, pickle, os, sys

FILES = [
    ("rss.dat", "rss.json"),
    ("classifierTrained.dat", "classifierTrained.json"),
]

for dat_path, json_path in FILES:
    if not os.path.exists(dat_path):
        print(f"[Skip] {dat_path} not found")
        continue
    with open(dat_path, "rb") as f:
        data = pickle.load(f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] {dat_path} -> {json_path}")
