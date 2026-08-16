#!/usr/bin/env python3
"""Calls a running catboost_model_server from the host and prints responses.

Prereqs:  pip install grpcio grpcio-tools
Server:   docker run -d --name catboost-server -p 8500:8500 \
              -v /tmp/cb-models:/models -v "$PWD/docker/e2e:/config" \
              catboost-serving:devel

Usage:
  python3 examples/predict_client.py                       # demo: all 3 fixture models
  python3 examples/predict_client.py --model mixed_multiclass \
      --numeric 0,0,0.1,-0.1 --categorical triangle,wood   # one custom row
"""
import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_DIR = REPO_ROOT / "examples" / "_gen"


def ensure_stubs():
    """Generates python stubs from the repo protos on first run."""
    if (GEN_DIR / "tfs" / "proto" / "catboost_predict_pb2_grpc.py").exists():
        return
    from grpc_tools import protoc
    # Well-known types (google/protobuf/*.proto) ship inside grpc_tools; the
    # programmatic API doesn't add them to the include path automatically.
    wkt_include = pathlib.Path(protoc.__file__).parent / "_proto"
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    rc = protoc.main([
        "protoc",
        f"-I{REPO_ROOT}",
        f"-I{REPO_ROOT}/third_party/tensorflow_serving_apis",
        f"-I{wkt_include}",
        f"--python_out={GEN_DIR}",
        f"--grpc_python_out={GEN_DIR}",
        # Canonical import paths (relative to the -I roots) — passing absolute
        # paths would make protoc see each file twice under two names.
        "tfs/proto/catboost_predict.proto",
        "tensorflow_serving/apis/model.proto",
    ])
    if rc != 0:
        sys.exit("stub generation failed — is grpcio-tools installed?")
    for pkg in ["tfs", "tfs/proto"]:
        (GEN_DIR / pkg / "__init__.py").touch()


def predict(stub, pb, model, numeric, categorical, probabilities):
    request = pb.CatBoostPredictRequest()
    request.model_spec.name = model
    request.output_probabilities = probabilities
    for i in range(max(len(numeric), len(categorical))):
        row = request.rows.add()
        if numeric:
            row.numeric.extend(numeric[i])
        if categorical:
            row.categorical.extend(categorical[i])
    return stub.Predict(request, timeout=10)


def show(model, kind, response):
    values = list(response.values)
    dim = response.dimension or 1
    rows = [values[i:i + dim] for i in range(0, len(values), dim)]
    print(f"  {kind:<13} (served version {response.model_spec.version.value}, "
          f"dimension {dim})")
    for i, row in enumerate(rows):
        print(f"    row {i}: {[round(v, 6) for v in row]}")


# proba=False for the regression model: the server would happily return
# sigmoid(raw), but that has no meaning for a regression target.
DEMO_ROWS = {
    "numeric_only": {
        "numeric": [[0, 0, 0, 0, 0], [1, -1, 0.5, 2, -0.3]],
        "categorical": [],
        "proba": False,
    },
    "categorical_only": {
        "numeric": [],
        "categorical": [["red", "XL", "city03"], ["blue", "S", "city05"]],
        "proba": True,
    },
    "mixed_multiclass": {
        "numeric": [[-3, 2, 0, 0], [0, 0, 0.1, -0.1], [3, -2, 0.5, 0.5]],
        "categorical": [["square", "metal"], ["triangle", "wood"],
                        ["circle", "plastic"]],
        "proba": True,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="localhost:8500")
    parser.add_argument("--model", help="model name; omit to demo all three")
    parser.add_argument("--numeric",
                        help="comma-separated floats for one row, e.g. 0,0,0.1,-0.1")
    parser.add_argument("--categorical",
                        help="comma-separated strings for one row, e.g. triangle,wood")
    args = parser.parse_args()

    ensure_stubs()
    sys.path.insert(0, str(GEN_DIR))
    import grpc
    from tfs.proto import catboost_predict_pb2 as pb
    from tfs.proto import catboost_predict_pb2_grpc as pb_grpc

    stub = pb_grpc.CatBoostPredictionServiceStub(
        grpc.insecure_channel(args.target))

    if args.model and (args.numeric or args.categorical):
        numeric = [[float(x) for x in args.numeric.split(",")]] if args.numeric else []
        categorical = [args.categorical.split(",")] if args.categorical else []
        targets = {args.model: {"numeric": numeric, "categorical": categorical,
                                "proba": True}}
    else:
        targets = DEMO_ROWS

    for model, rows in targets.items():
        print(f"\n{model}  @ {args.target}")
        try:
            show(model, "raw values",
                 predict(stub, pb, model, rows["numeric"], rows["categorical"], False))
            if rows["proba"]:
                show(model, "probabilities",
                     predict(stub, pb, model, rows["numeric"], rows["categorical"], True))
        except grpc.RpcError as e:
            print(f"  RPC failed: {e.code().name} — {e.details()}")


if __name__ == "__main__":
    main()
