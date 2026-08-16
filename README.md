# catboost-serving

Serving CatBoost models (`.cbm`) with TensorFlow Serving: a new TFS *platform*
named `catboost`, with full TFS version management (hot reload of
`/models/<name>/<version>/`), a dedicated gRPC API, and unit/e2e tests against
genuinely trained models. See [docs/SPEC.md](docs/SPEC.md) for the full design.

Supports numerical-only, categorical-only and mixed feature models, single-
and multi-target (multiclass) outputs. Categorical features are passed as raw
strings (CatBoost hashes internally, so unseen categories behave exactly like
in Python). v1 non-goals: REST, text/embedding features, GPU, batching.

## Layout

| Path | What | Builds where |
|---|---|---|
| `cb/` | **L1** — `cb::CatBoostModel` RAII wrapper over the CatBoost C API + the main unit test | natively on macOS arm64 and Linux (own Bazel module, no TF deps) |
| `third_party/catboost/` | vendored C API headers (v1.2.8) + `cc_import` of the prebuilt `libcatboostmodel` (pinned by sha256 in `MODULE.bazel`) | both |
| `tfs/` | **L2** — TFS source adapter, gRPC service, server `main.cc`, protos. `tfs/BUILD` is written for the TFS tree (see below) and the dir is `.bazelignore`d in L1 | only inside the TFS devel Docker image |
| `docker/` | `Dockerfile.devel` (L2 build) + e2e smoke test harness | — |

## L1: dev loop on the Mac (and Linux)

```sh
bazel test //cb:catboost_model_test        # 7 tests, all golden-value based
```

Regenerate the test fixtures (only if you change the training script;
commit the results — goldens and models must stay in sync):

```sh
pip install catboost numpy
python3 cb/testdata/train_test_models.py
```

Verify L1 on Linux without leaving the Mac:

```sh
docker run --rm --platform linux/arm64 -v "$PWD:/w" -w /w ubuntu:24.04 bash -c '
  apt-get update -qq && apt-get install -y -qq curl g++ python3 zip unzip git &&
  curl -sfL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64 &&
  chmod +x /usr/local/bin/bazel && bazel test //cb:catboost_model_test'
```

## L2: build the server (Docker, linux/amd64)

The TFS devel image ships with its Bazel cache wiped, so the first build
recompiles the TF dependency tree (**hours** under Rosetta on Apple Silicon;
`--jobs=6 --local_ram_resources=12288` keeps a 24 GB machine alive). It is a
one-time cost — the prewarm layer is cached, and iterating on
`catboost_serving` sources afterwards takes minutes.

```sh
docker build --platform linux/amd64 -f docker/Dockerfile.devel -t catboost-serving:devel .
```

This also runs `catboost_source_adapter_test` inside the image.

How the L2 build wires into TFS 2.20 (all additive, no patches to
`tensorflow_serving/`):

- sources are copied to `tensorflow_serving/catboost_serving/` inside the TFS
  tree (must live under `//tensorflow_serving/...` — most TFS deps are
  visibility-restricted to that subtree); `tfs/BUILD` becomes the package
  BUILD and uses `includes = ["."]` so include paths match L1;
- `docker/workspace_append.txt` is appended to the TFS `WORKSPACE` to define
  the `libcatboostmodel.so` downloads;
- the adapter library is built `alwayslink = 1` — its static initializer
  registers the `catboost` platform.

## Run the server

Model repository layout is standard TFS: `<base_path>/<version>/model.cbm`
(exactly one `.cbm` per version dir).

```sh
docker run --rm -p 8500:8500 \
  -v "$PWD/my_models:/models" -v "$PWD/docker/e2e:/config" \
  catboost-serving:devel \
  --port=8500 --model_config_file=/config/models.conf \
  --platform_config_file=/config/platform.conf
```

See [docker/e2e/models.conf](docker/e2e/models.conf) and
[docker/e2e/platform.conf](docker/e2e/platform.conf) for the two config files.
Dropping a new `/models/<name>/<N+1>/model.cbm` hot-swaps the version within
`--file_system_poll_wait_seconds` (default 1s), with availability preserved.

Example call with `grpcurl`:

```sh
grpcurl -plaintext \
  -import-path . -import-path third_party/tensorflow_serving_apis \
  -proto tfs/proto/catboost_predict.proto \
  -d '{"model_spec":{"name":"mixed_multiclass"},
       "rows":[{"numeric":[0,0,0.1,-0.1],"categorical":["triangle","wood"]}],
       "output_probabilities":true}' \
  localhost:8500 catboost.serving.CatBoostPredictionService/Predict
```

For binary models the probability output is one value per row — P(class 1),
i.e. sigmoid of the raw logit; multiclass returns a softmax row per document.

## End-to-end smoke test

Serves all three fixture models, checks gRPC predictions against the Python
golden values (rtol 1e-6, incl. multiclass probabilities), then copies
`mixed_multiclass` to version 2 and asserts the echoed version flips without
a restart:

```sh
./docker/e2e/run_e2e.sh          # uses image catboost-serving:devel
```

## Pinned versions

- CatBoost **1.2.8** (`libcatboostmodel` binaries by sha256 + vendored C API headers — keep in sync)
- TensorFlow Serving **2.20.0** (latest release; `tensorflow/serving:2.20.0-devel`, amd64-only)
- Bazel **9.2.0** for L1 (`.bazelversion`); L2 uses whatever Bazel the TFS devel image pins
