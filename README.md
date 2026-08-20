# tfserving-catboost

Serving CatBoost models (`.cbm`) with TensorFlow Serving: a new TFS *platform*
named `catboost`, with full TFS version management (hot reload of
`/models/<name>/<version>/`), a dedicated gRPC API, and unit/e2e tests against
genuinely trained models. See [docs/SPEC.md](docs/SPEC.md) for the design and
[docs/TEST_REPORT.md](docs/TEST_REPORT.md) for what was tested.

Supports numerical-only, categorical-only and mixed feature models, single-
and multi-target (multiclass) outputs. Categorical features are passed as raw
strings (CatBoost hashes internally, so unseen categories behave exactly like
in Python). v1 non-goals: REST, text/embedding features, GPU, batching.

## Platform support at a glance

| | L1 (wrapper + unit tests) | L2 (Docker image) + e2e | Serving in production |
|---|---|---|---|
| **macOS Apple Silicon** | ✅ native (tested) | ✅ via Rosetta emulation — slow first build | dev only |
| **macOS Intel** | ✅ native | ✅ native amd64 containers | dev only |
| **Linux x86_64** | ✅ native (tested) | ✅ native — reference platform (tested) | ✅ recommended |
| **Linux aarch64** | ✅ native (tested) | ❌ upstream TFS images are amd64-only | ❌ |
| **Windows 10/11 + WSL2** | ✅ inside WSL2 (as Linux x86_64) | ✅ Docker Desktop, WSL2 backend | via Linux hosts |
| **Windows native** | ⚠️ wired (DLL + MSVC branch), untested | ❌ use WSL2 | ❌ |

The same checkout works everywhere — the Bazel build selects the right
prebuilt `libcatboostmodel` (dylib / .so / .dll) per OS and CPU automatically.

## Layout

| Path | What | Builds where |
|---|---|---|
| `cb/` | **L1** — `cb::CatBoostModel` RAII wrapper over the CatBoost C API + the main unit test | natively on your machine (own Bazel module, no TF deps) |
| `third_party/catboost/` | vendored C API headers (v1.2.8) + `cc_import` of the prebuilt `libcatboostmodel` (pinned by sha256 in `MODULE.bazel`) | all platforms |
| `tfs/` | **L2** — TFS source adapter, gRPC service, server `main.cc`, protos. `tfs/BUILD` is written for the TFS tree and the dir is `.bazelignore`d in L1 | only inside the TFS devel Docker image |
| `docker/` | `Dockerfile.devel` (L2 build) + e2e smoke test harness | — |
| `examples/` | Python gRPC client (auto-generates its stubs) | all platforms |
| `ci/` | GitHub Actions definition (see [Continuous integration](#continuous-integration)) | — |

## Getting started, per platform

Everything needs [Bazelisk](https://github.com/bazelbuild/bazelisk) (a `bazel`
launcher that reads `.bazelversion`), Python 3, and — for L2/e2e — Docker.

### macOS (Apple Silicon or Intel)

```sh
brew install bazelisk grpcurl        # grpcurl optional, for ad-hoc calls
bazel test //cb:catboost_model_test  # L1: the 7 golden-value tests, native
```

Apple Silicon notes: the L1 build uses the universal2 dylib natively (arm64);
the **L2 image build runs under Rosetta** because upstream
`tensorflow/serving` images are amd64-only — the first build compiles the TF
dependency tree (hours; the Dockerfile caps Bazel at `--jobs=6
--local_ram_resources=12288`, sized for a 24 GB machine — raise both in
`docker/Dockerfile.devel` ARGs if you have more). Intel Macs run the same
build natively, no emulation.

Optional cross-check of the Linux build without leaving the Mac:

```sh
# pick the platform/bazelisk arch matching what you want to check:
#   --platform linux/arm64  + bazelisk-linux-arm64   (Apple Silicon, fast)
#   --platform linux/amd64  + bazelisk-linux-amd64   (via Rosetta)
docker run --rm --platform linux/arm64 -v "$PWD:/w" -w /w ubuntu:24.04 bash -c '
  apt-get update -qq && apt-get install -y -qq curl g++ python3 zip unzip git &&
  curl -sfL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64 &&
  chmod +x /usr/local/bin/bazel && bazel test //cb:catboost_model_test'
```

### Linux (x86_64 — the reference platform)

```sh
# bazelisk: grab the binary once
sudo curl -sfL -o /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
  && sudo chmod +x /usr/local/bin/bazel

bazel test //cb:catboost_model_test   # L1, native
```

L2 builds natively (no `--platform` flag needed, and you can raise the Bazel
job/RAM caps to whatever your machine has). aarch64 hosts: L1 works natively;
the L2 image cannot be built there (amd64-only upstream base — an arm64 image
would require building TFS from source).

### Windows

**Recommended: WSL2.** Install a WSL2 distro (e.g. `wsl --install -d Ubuntu`),
plus Docker Desktop with the WSL2 backend, then follow the **Linux** section
inside WSL verbatim — everything (L1, L2 image, e2e, client) behaves exactly
as on Linux x86_64, with containers running natively.

**Native Windows (L1 only, currently untested):** the build is wired — the
Bazel `select()` imports the official `catboostmodel.dll` + import `.lib`, and
`.bazelrc` passes MSVC-style flags — but no Windows machine has run it yet.
You need the MSVC Build Tools (C++ workload) and
[bazelisk.exe](https://github.com/bazelbuild/bazelisk/releases); then, from a
*x64 Native Tools* prompt:

```bat
bazel test //cb:catboost_model_test
```

The Python client and fixture regeneration also work natively (plain
`pip install grpcio grpcio-tools catboost numpy`). The e2e harness is a bash
script — run it from WSL or Git Bash. If you try native Windows, issues/PRs
welcome.

## L1: the dev loop (all platforms)

```sh
bazel test //cb:catboost_model_test        # 7 tests, all golden-value based
```

Regenerate the test fixtures (only if you change the training script;
commit the results — goldens and models must stay in sync):

```sh
pip install catboost numpy
python3 cb/testdata/train_test_models.py
```

Training is seeded, so the *predictions* it writes are reproducible bit-for-bit
— rerunning the script leaves `expected_predictions.json` byte-identical. The
`.cbm` files themselves are not byte-reproducible (CatBoost embeds
non-deterministic metadata), so `git status` will show them modified even when
nothing about the models changed; discard those with `git checkout` unless you
actually retrained.

## L2: build the server image (Docker, linux/amd64)

The TFS devel image ships with its Bazel cache wiped, so the first build
recompiles the TF dependency tree (**hours**; one-time — the prewarm layer is
cached and iterating on `catboost_serving` sources afterwards takes minutes).

```sh
docker build --platform linux/amd64 -f docker/Dockerfile.devel -t catboost-serving:devel .
```

This also runs `catboost_source_adapter_test` inside the image. The
`--platform linux/amd64` flag matters only on arm64 hosts (Apple Silicon);
it's a no-op on x86_64 Linux/Windows/Intel-Mac.

How the L2 build wires into TFS 2.20 (all additive, no patches to
`tensorflow_serving/`):

- sources are copied to `tensorflow_serving/catboost_serving/` inside the TFS
  tree (must live under `//tensorflow_serving/...` — most TFS deps are
  visibility-restricted to that subtree); `tfs/BUILD` becomes the package
  BUILD and uses `includes = ["."]` so include paths match L1;
- `docker/workspace_append.txt` is appended to the TFS `WORKSPACE` to define
  the `libcatboostmodel.so` downloads;
- the adapter library is built `alwayslink = 1` — its static initializer
  registers the `catboost` platform;
- the binary's `libcatboostmodel.so` is installed system-wide in the image
  (Bazel's rpath breaks once the binary leaves `bazel-bin`).

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

## Call it

Python client (all platforms; generates its own stubs on first run):

```sh
pip install grpcio grpcio-tools
python3 examples/predict_client.py                     # demo rows, all 3 models
python3 examples/predict_client.py --model mixed_multiclass \
    --numeric 0,0,0.1,-0.1 --categorical triangle,wood
```

Or `grpcurl`:

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
a restart. macOS / Linux / WSL:

```sh
./docker/e2e/run_e2e.sh          # uses image catboost-serving:devel
```

## Continuous integration

The GitHub Actions definition lives at
[ci/github-actions-ci.yml](ci/github-actions-ci.yml) rather than under
`.github/workflows/` — pushing to that path requires a token with the
`workflow` scope. To activate it:

```sh
gh auth refresh -h github.com -s workflow      # one-time scope grant
mkdir -p .github/workflows
git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "Activate GitHub Actions CI" && git push
```

It runs the L1 test suite on ubuntu-24.04 (the authoritative Linux x86_64
check) on every push, and can build the L2 image on manual dispatch.

## Pinned versions

- CatBoost **1.2.8** (`libcatboostmodel` binaries by sha256 for macOS
  universal2, Linux x86_64/aarch64, Windows x86_64 + vendored C API headers —
  keep in sync)
- TensorFlow Serving **2.20.0** (latest release; `tensorflow/serving:2.20.0-devel`, amd64-only)
- Bazel **9.2.0** for L1 (`.bazelversion`, via bazelisk); L2 uses whatever
  Bazel the TFS devel image pins
