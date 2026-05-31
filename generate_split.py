import os
import random

PREPROCESS_DIR = "preprocess_120_betaFalse"
OUT_DIR = os.environ.get("TRUMANS_DATA_ROOT", "/home/dataset/xingyu/trumans/Data_release")
NUM_FRAMES = 196
SEED = 42

random.seed(SEED)

REQUIRED = ["body_parms_cano_gt.pickle", "scene_info.pickle", "occ_scene.npz"]

entries = []
total = 0
missing_count = {f: 0 for f in REQUIRED}
incomplete_samples = []  # (sample_path, [missing files])

for seg in sorted(os.listdir(PREPROCESS_DIR)):
    seg_path = os.path.join(PREPROCESS_DIR, seg)
    if not os.path.isdir(seg_path):
        continue
    for seg_idx in sorted(os.listdir(seg_path)):
        sub = os.path.join(seg_path, seg_idx)
        if not os.path.isdir(sub):
            continue
        total += 1
        missing = [f for f in REQUIRED if not os.path.exists(os.path.join(sub, f))]
        if missing:
            for f in missing:
                missing_count[f] += 1
            incomplete_samples.append((f"{seg}/{seg_idx}", missing))
        else:
            entries.append(f"a/b/{seg}/{seg_idx}")

print(f"\n=== Summary ===")
print(f"Total samples scanned : {total}")
print(f"Complete samples      : {len(entries)}")
print(f"Incomplete samples    : {len(incomplete_samples)}")
print(f"\nMissing file counts:")
for f, c in missing_count.items():
    print(f"  {f:35s} missing in {c} samples")

if incomplete_samples:
    print(f"\n=== Incomplete samples (showing all {len(incomplete_samples)}) ===")
    for sample, miss in incomplete_samples:
        print(f"  {sample}: missing {miss}")
print()

random.shuffle(entries)
n = len(entries)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    "train": entries[:n_train],
    "val":   entries[n_train:n_train + n_val],
    "test":  entries[n_train + n_val:],
}

os.makedirs(OUT_DIR, exist_ok=True)
for name, lst in splits.items():
    out_path = os.path.join(OUT_DIR, f"{name}_{NUM_FRAMES}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lst))
    print(f"  {out_path}: {len(lst)} samples")
