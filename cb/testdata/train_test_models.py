#!/usr/bin/env python3
"""Generates the three .cbm test fixtures and expected_predictions.json.

Run once (pip install catboost numpy), commit outputs:
    python3 cb/testdata/train_test_models.py

The golden JSON stores, per model, 8 fixed input rows and the Python-side
outputs of predict(prediction_type='RawFormulaVal') and (for classifiers)
predict_proba. The C++ wrapper must reproduce these within rtol 1e-6.
"""
import json
import os

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
rng = np.random.default_rng(SEED)


def save(model, name):
    path = os.path.join(OUT_DIR, name)
    model.save_model(path, format="cbm")
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


# ---------------------------------------------------------------- numeric_only
# Regression, 5 float features, y = 3*x0 - 2*x1 + x2*x3 + noise.
N = 2000
Xn = rng.normal(0, 1, size=(N, 5)).astype(np.float32)
yn = 3 * Xn[:, 0] - 2 * Xn[:, 1] + Xn[:, 2] * Xn[:, 3] + rng.normal(0, 0.1, N)

numeric_model = CatBoostRegressor(
    iterations=50, depth=4, random_seed=SEED, verbose=False, allow_writing_files=False
)
numeric_model.fit(Xn, yn)
save(numeric_model, "numeric_only.cbm")

# 8 golden rows: ordinary values plus extremes.
numeric_rows = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.5, 2.0, -0.3],
    [-2.5, 3.1, 0.0, 1.0, 1.0],
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [1e6, -1e6, 1e6, -1e6, 1e6],       # extreme values
    [-1e-8, 1e-8, 0.0, -0.0, 42.0],
    [5.0, 5.0, 5.0, 5.0, 5.0],
    [-3.0, -3.0, 3.0, 3.0, 0.0],
]

# ----------------------------------------------------------- categorical_only
# Binary classification, 3 cat features; label depends on category combos.
colors = ["red", "green", "blue"]
sizes = ["S", "M", "L", "XL"]
cities = [f"city{i:02d}" for i in range(20)]

Nc = 3000
cat_data = np.column_stack([
    rng.choice(colors, Nc),
    rng.choice(sizes, Nc),
    rng.choice(cities, Nc),
])
yc = (
    (cat_data[:, 0] == "red") & np.isin(cat_data[:, 1], ["L", "XL"])
    | (cat_data[:, 2] < "city10") & (cat_data[:, 0] == "blue")
).astype(int)
# flip a few labels for noise
flip = rng.random(Nc) < 0.05
yc = np.where(flip, 1 - yc, yc)

cat_model = CatBoostClassifier(
    iterations=60, depth=4, random_seed=SEED, verbose=False, allow_writing_files=False
)
cat_model.fit(Pool(cat_data.tolist(), yc.tolist(), cat_features=[0, 1, 2]))
save(cat_model, "categorical_only.cbm")

categorical_rows = [
    ["red", "XL", "city03"],
    ["blue", "S", "city05"],
    ["green", "M", "city15"],
    ["red", "S", "city19"],
    ["blue", "L", "city00"],
    ["purple", "XXL", "atlantis"],   # categories never seen in training
    ["green", "XL", "city10"],
    ["red", "L", "city07"],
]

# ----------------------------------------------------------- mixed_multiclass
# 3-class classification, 4 float + 2 cat features.
shapes = ["circle", "square", "triangle"]
materials = ["wood", "metal", "plastic", "glass"]

Nm = 3000
Xm_num = rng.normal(0, 1, size=(Nm, 4)).astype(np.float32)
Xm_cat = np.column_stack([rng.choice(shapes, Nm), rng.choice(materials, Nm)])
score = (
    2 * Xm_num[:, 0]
    - Xm_num[:, 1]
    + (Xm_cat[:, 0] == "circle") * 2.0
    + (Xm_cat[:, 1] == "metal") * -1.5
    + rng.normal(0, 0.5, Nm)
)
ym = np.digitize(score, [-1.0, 1.5])  # 3 classes: 0, 1, 2

mixed_rows_py = [list(Xm_num[i]) + list(Xm_cat[i]) for i in range(Nm)]
mixed_model = CatBoostClassifier(
    iterations=60, depth=4, loss_function="MultiClass",
    random_seed=SEED, verbose=False, allow_writing_files=False,
)
mixed_model.fit(Pool(mixed_rows_py, ym.tolist(), cat_features=[4, 5]))
save(mixed_model, "mixed_multiclass.cbm")

# 8 golden rows: one per class regime, extremes, unseen category.
mixed_float_rows = [
    [-3.0, 2.0, 0.0, 0.0],    # strongly class 0
    [0.0, 0.0, 0.1, -0.1],    # middle -> class 1
    [3.0, -2.0, 0.5, 0.5],    # strongly class 2
    [1.0, 1.0, 1.0, 1.0],
    [-0.5, 0.5, -1.0, 2.0],
    [1e5, -1e5, 0.0, 1.0],    # extreme floats
    [0.2, -0.3, 0.4, -0.5],
    [2.0, 0.0, -2.0, 0.0],
]
mixed_cat_rows = [
    ["square", "metal"],
    ["triangle", "wood"],
    ["circle", "plastic"],
    ["circle", "metal"],
    ["square", "glass"],
    ["triangle", "plastic"],
    ["hexagon", "carbon"],    # unseen categories
    ["circle", "wood"],
]


def golden_entry(model, float_rows, cat_rows, cat_indices, classifier):
    """Computes raw + proba references using the exact row layout CatBoost saw."""
    if float_rows and cat_rows:
        rows = [list(map(float, f)) + list(c) for f, c in zip(float_rows, cat_rows)]
    elif float_rows:
        rows = [list(map(float, f)) for f in float_rows]
    else:
        rows = [list(c) for c in cat_rows]
    pool = Pool(rows, cat_features=cat_indices)
    raw = model.predict(pool, prediction_type="RawFormulaVal")
    entry = {
        "float_features": [list(map(float, f)) for f in float_rows] if float_rows else [],
        "cat_features": cat_rows if cat_rows else [],
        "raw": np.asarray(raw, dtype=np.float64).reshape(len(rows), -1).tolist(),
    }
    if classifier:
        proba = model.predict_proba(pool)
        entry["proba"] = np.asarray(proba, dtype=np.float64).tolist()
    return entry


golden = {
    "numeric_only": golden_entry(numeric_model, numeric_rows, [], [], classifier=False),
    "categorical_only": golden_entry(cat_model, [], categorical_rows, [0, 1, 2], classifier=True),
    "mixed_multiclass": golden_entry(
        mixed_model, mixed_float_rows, mixed_cat_rows, [4, 5], classifier=True
    ),
}

json_path = os.path.join(OUT_DIR, "expected_predictions.json")
with open(json_path, "w") as f:
    json.dump(golden, f, indent=1)
print(f"wrote {json_path}")

for name, e in golden.items():
    dim = len(e["raw"][0])
    print(f"{name}: 8 rows, dimension={dim}, raw[0]={e['raw'][0]}")
