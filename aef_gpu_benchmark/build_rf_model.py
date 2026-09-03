import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

base = Path(__file__).parent
raw = np.fromfile(base / "overview_1024x1024x64.raw", dtype=np.int8).reshape(-1, 64)
x = raw.astype(np.float32)
m = np.abs(x) / 127.5
x = np.sign(x) * m * m
x[raw == -128] = 0
rng = np.random.default_rng(7)
train_idx = rng.choice(len(x), size=50000, replace=False)
train_x = x[train_idx]
classification_y = ((train_x[:, 7] + 0.7 * train_x[:, 31] - 0.25 * train_x[:, 48]) > 0).astype(np.int32)
regression_y = 3.0 * train_x[:, 3] - 1.5 * train_x[:, 22] + 0.8 * train_x[:, 57] + rng.normal(0, 0.03, len(train_x))

models = {
    "classification": RandomForestClassifier(n_estimators=256, max_depth=18, min_samples_leaf=2, random_state=11, n_jobs=-1).fit(train_x, classification_y),
    "regression": RandomForestRegressor(n_estimators=256, max_depth=18, min_samples_leaf=2, random_state=11, n_jobs=-1).fit(train_x, regression_y),
}

def export(model, name):
    offsets = []
    features = []
    thresholds = []
    left = []
    right = []
    leaves = []
    for tree in model.estimators_:
        t = tree.tree_
        start = len(left)
        offsets.append(start)
        for i in range(t.node_count):
            features.append(int(t.feature[i]) if t.feature[i] >= 0 else 0)
            if t.children_left[i] < 0:
                left.append(-1); right.append(-1)
                if name == "classification":
                    counts = t.value[i, 0]
                    leaves.append(float(np.argmax(counts)))
                else:
                    leaves.append(float(t.value[i, 0, 0]))
            else:
                # Convert f(q) <= threshold into q <= cutoff. The mapping is monotonic.
                threshold = float(t.threshold[i])
                candidates = np.arange(-127, 128, dtype=np.float32)
                values = np.sign(candidates) * (np.abs(candidates) / 127.5) ** 2
                cutoff = int(candidates[values <= threshold].max()) if np.any(values <= threshold) else -128
                left.append(start + int(t.children_left[i]))
                right.append(start + int(t.children_right[i]))
                leaves.append(0.0)
            thresholds.append(cutoff if t.children_left[i] >= 0 else 0)
    out = base / f"rf_{name}"
    np.asarray(offsets, np.int32).tofile(out.with_suffix(".offsets.i32"))
    np.asarray(features, np.int16).tofile(out.with_suffix(".features.i16"))
    np.asarray(thresholds, np.int16).tofile(out.with_suffix(".thresholds.i16"))
    np.asarray(features, np.uint8).tofile(out.with_suffix(".features.u8"))
    np.asarray(thresholds, np.int8).tofile(out.with_suffix(".thresholds.i8"))
    np.asarray(left, np.int32).tofile(out.with_suffix(".left.i32"))
    np.asarray(right, np.int32).tofile(out.with_suffix(".right.i32"))
    np.asarray(leaves, np.float32).tofile(out.with_suffix(".leaves.f32"))
    with out.with_suffix(".json").open("w") as f:
        json.dump({"name": name, "tree_count": len(offsets), "node_count": len(left), "max_depth": 18}, f, indent=2)

for name, model in models.items():
    export(model, name)
    joblib.dump(model, base / f"rf_{name}.joblib")
    pred = model.predict(x)
    pred.astype(np.float32).tofile(base / f"rf_{name}.cpu.f32")
print({name: {"trees": len(model.estimators_), "nodes": int(sum(t.tree_.node_count for t in model.estimators_))} for name, model in models.items()})
