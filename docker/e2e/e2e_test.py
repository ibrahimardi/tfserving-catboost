#!/usr/bin/env python3
"""End-to-end smoke test against a running catboost_model_server.

Prereqs (run_e2e.sh does all of this):
  - server running with docker/e2e/{models.conf,platform.conf} and the three
    fixture models mounted at /models/<name>/1/model.cbm
  - python stubs generated into --stub_dir via grpc_tools.protoc
  - the writable models dir passed as --models_dir (for the hot-swap check)

Asserts, per model: raw values and probabilities match
cb/testdata/expected_predictions.json (rtol 1e-6); then copies
mixed_multiclass to version 2 and verifies the echoed version flips to 2
without a restart.
"""
import argparse
import json
import os
import shutil
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--target", default="localhost:8500")
parser.add_argument("--stub_dir", required=True)
parser.add_argument("--models_dir", required=True)
parser.add_argument("--golden", required=True,
                    help="path to expected_predictions.json")
args = parser.parse_args()

sys.path.insert(0, args.stub_dir)

import grpc  # noqa: E402
from tfs.proto import catboost_predict_pb2 as pb  # noqa: E402
from tfs.proto import catboost_predict_pb2_grpc as pb_grpc  # noqa: E402

RTOL = 1e-6
with open(args.golden) as f:
    GOLDEN = json.load(f)

channel = grpc.insecure_channel(args.target)
stub = pb_grpc.CatBoostPredictionServiceStub(channel)

failures = []


def check(name, actual, expected, context):
    ok = len(actual) == len(expected) and all(
        abs(a - e) <= RTOL * max(1.0, abs(e)) for a, e in zip(actual, expected)
    )
    if not ok:
        failures.append(f"{name} {context}: got {actual}, want {expected}")


def make_request(entry, model, probabilities):
    req = pb.CatBoostPredictRequest()
    req.model_spec.name = model
    req.output_probabilities = probabilities
    n_rows = max(len(entry["float_features"]), len(entry["cat_features"]))
    for i in range(n_rows):
        row = req.rows.add()
        if entry["float_features"]:
            row.numeric.extend(entry["float_features"][i])
        if entry["cat_features"]:
            row.categorical.extend(entry["cat_features"][i])
    return req


def wait_ready(model, deadline_s=60):
    req = pb.CatBoostPredictRequest()
    req.model_spec.name = model
    row = req.rows.add()
    entry = GOLDEN[model]
    if entry["float_features"]:
        row.numeric.extend(entry["float_features"][0])
    if entry["cat_features"]:
        row.categorical.extend(entry["cat_features"][0])
    start = time.time()
    while time.time() - start < deadline_s:
        try:
            stub.Predict(req, timeout=5)
            return
        except grpc.RpcError:
            time.sleep(1)
    raise RuntimeError(f"{model} not available after {deadline_s}s")


for model, entry in GOLDEN.items():
    wait_ready(model)

    # Raw values.
    resp = stub.Predict(make_request(entry, model, probabilities=False))
    golden_raw = [v for row in entry["raw"] for v in row]
    check(model, list(resp.values), golden_raw, "raw")
    dim = len(entry["raw"][0])
    if resp.dimension != dim:
        failures.append(f"{model}: dimension {resp.dimension}, want {dim}")
    if resp.model_spec.version.value != 1:
        failures.append(
            f"{model}: version {resp.model_spec.version.value}, want 1")

    # Probabilities. The server returns sigmoid (P of class 1, one value/row)
    # for binary models and softmax rows for multiclass.
    if "proba" in entry:
        resp = stub.Predict(make_request(entry, model, probabilities=True))
        n_classes = len(entry["proba"][0])
        if n_classes == 2:
            golden_proba = [row[1] for row in entry["proba"]]
        else:
            golden_proba = [v for row in entry["proba"] for v in row]
        check(model, list(resp.values), golden_proba, "proba")
    print(f"OK {model}")

# Hot version swap: copy mixed_multiclass to version 2, expect the echoed
# version to flip without restart.
src = os.path.join(args.models_dir, "mixed_multiclass", "1", "model.cbm")
dst_dir = os.path.join(args.models_dir, "mixed_multiclass", "2")
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, os.path.join(dst_dir, "model.cbm"))

entry = GOLDEN["mixed_multiclass"]
deadline = time.time() + 60
version = None
while time.time() < deadline:
    resp = stub.Predict(make_request(entry, "mixed_multiclass", False))
    version = resp.model_spec.version.value
    if version == 2:
        break
    time.sleep(1)
if version != 2:
    failures.append(f"hot swap: version stuck at {version}, want 2")
else:
    golden_raw = [v for row in entry["raw"] for v in row]
    check("mixed_multiclass@2", list(resp.values), golden_raw, "raw")
    print("OK hot version swap 1 -> 2")

if failures:
    print("\nE2E FAILURES:")
    for failure in failures:
        print(" -", failure)
    sys.exit(1)
print("\nE2E PASSED")
