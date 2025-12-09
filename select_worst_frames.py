# --------------------------------------------------------------
# File: select_worst_frames.py
# --------------------------------------------------------------
import json, os




def select_topk_from_scores(scores_json, k, pool_dir, target_dir):
with open(scores_json) as f:
scores = json.load(f)
items = []
for kf,v in scores.items():
score = v.get('E_var',0.0) + v.get('F_var',0.0)
items.append((kf, score))
items.sort(key=lambda x: x[1], reverse=True)
selected = [i[0] for i in items[:k]]
os.makedirs(target_dir, exist_ok=True)
for s in selected:
src = os.path.join(pool_dir, s)
dst = os.path.join(target_dir, s)
if os.path.exists(src):
os.system(f'cp {src} {dst}')
return selected
