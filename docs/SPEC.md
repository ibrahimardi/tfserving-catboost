# Serving CatBoost Models with TensorFlow Serving — Implementation Spec

**Goal:** Extend TensorFlow Serving (TFS) with a new *platform* that serves CatBoost models (`.cbm`) — supporting **numerical-only, categorical-only, and mixed** feature models, single- and multi-target (multiclass) outputs — with full TFS version management (hot reload of `/models/<name>/<version>/`), gRPC API, and a real unit test against a genuinely trained model.

**Non-goal (v1):** REST API, text/embedding features, GPU inference, batching scheduler integration.

---

## 0. Dev environment (Mac M4 Pro / 24 GB — must also build on Linux)

TFS **does not build natively on macOS**. The architecture below is split into two layers precisely so most dev/test iteration happens natively on the Mac:

| Layer | Depends on | Where it builds & tests |
|---|---|---|
| **L1: `catboost_model` wrapper + unit tests** | Only `libcatboostmodel` + Abseil + GoogleTest (no TF, no TFS) | Natively on macOS arm64 **and** Linux (x86_64/aarch64) |
| **L2: TFS integration** (source adapter, gRPC service, server binary) | Full TFS tree (Bazel) | **Only inside Docker** (Linux) |

Rules:

1. **L1 is a standalone Bazel module** (own `MODULE.bazel`), no TFS deps. All prediction-correctness tests live here. Iterate here 90% of the time.
2. **L2 builds in Docker** using the `tensorflow/serving:latest-devel` image. On the M4:
   - Preferred: build for **linux/arm64** if a devel image/toolchain is available for aarch64; otherwise run **linux/amd64 under Rosetta 2** (`docker run --platform linux/amd64 ...`). amd64 emulation works but a full TFS build is slow (hours first time) and memory-hungry — cap Bazel with `--local_ram_resources=12288 --jobs=6` given 24 GB.
   - CI (GitHub Actions, ubuntu-24.04) does the authoritative Linux x86_64 build of L2.
3. `libcatboostmodel` prebuilt binaries: download from CatBoost GitHub releases (pin one version, e.g. the latest 1.2.x) — `libcatboostmodel.dylib` (darwin universal/arm64), `libcatboostmodel.so` (linux x86_64 and aarch64). Header: `catboost/libs/model_interface/model_calcer_wrapper.h` (C API — this is the ABI-stable interface; do **not** link the C++ `TFullModel` internals). Wrap per-platform binaries with a Bazel `select()` over `@platforms//os` + cpu.

---

## 1. Repository layout

```
catboost_serving/
├── MODULE.bazel                      # L1 standalone module
├── third_party/catboost/BUILD        # cc_import of libcatboostmodel per-platform (select())
├── cb/                               # ---- L1: no TF dependencies ----
│   ├── catboost_model.h / .cc        # RAII wrapper around C API
│   ├── catboost_model_test.cc        # THE unit test (section 5)
│   └── testdata/
│       ├── train_test_models.py      # generates the 3 .cbm fixtures + expected outputs
│       ├── numeric_only.cbm
│       ├── categorical_only.cbm
│       ├── mixed_multiclass.cbm
│       └── expected_predictions.json # inputs + reference outputs from Python CatBoost
├── tfs/                              # ---- L2: built inside TFS Docker ----
│   ├── proto/
│   │   ├── catboost_source_adapter.proto
│   │   └── catboost_predict.proto
│   ├── catboost_source_adapter.h / .cc
│   ├── catboost_source_adapter_test.cc
│   ├── catboost_service_impl.h / .cc
│   ├── main.cc                       # custom model server binary
│   └── BUILD
├── docker/Dockerfile.devel           # FROM tensorflow/serving:*-devel + libcatboostmodel.so
└── docs/  (this file)
```

---

## 2. L1 — the servable: `cb::CatBoostModel`

RAII wrapper over the CatBoost C API. Key API calls (all from `model_calcer_wrapper.h`):

- `ModelCalcerCreate()` / `ModelCalcerDelete()`
- `LoadFullModelFromFile(handle, path)` — returns `false` on failure; get message via `GetErrorString()`
- Introspection (drives validation): `GetFloatFeaturesCount()`, `GetCatFeaturesCount()`, `GetDimensionsCount()` (=1 for regression/binary raw value; =K for K-class multiclass)
- Prediction (covers all three model types with one call):
  ```c
  bool CalcModelPrediction(
      ModelCalcerHandle*, size_t docCount,
      const float** floatFeatures, size_t floatFeaturesSize,
      const char*** catFeatures,  size_t catFeaturesSize,
      double* result, size_t resultSize);
  ```
  Categorical features are passed **as raw strings** — CatBoost hashes internally, so string categories unseen at training time are handled the same way Python does. Numeric-only models: pass `catFeatures=nullptr, catFeaturesSize=0`; categorical-only: `floatFeatures=nullptr, floatFeaturesSize=0`.

Public interface:

```cpp
namespace cb {
class CatBoostModel {
 public:
  static absl::StatusOr<std::unique_ptr<CatBoostModel>> Load(const std::string& path);
  int float_feature_count() const;
  int cat_feature_count() const;
  int dimension() const;            // outputs per document
  // rows: batch. Each inner vector sized exactly to the model's feature counts.
  // Returns row-major [docCount x dimension] raw values.
  absl::StatusOr<std::vector<double>> Predict(
      const std::vector<std::vector<float>>& float_features,
      const std::vector<std::vector<std::string>>& cat_features) const;
  // Convenience: sigmoid/softmax applied for probability output.
  absl::StatusOr<std::vector<double>> PredictProba(...same args...) const;
};
}  // namespace cb
```

Implementation requirements:

- **Validate shapes before calling C API**: batch sizes of the two feature blocks must agree (one may be empty iff the model has 0 features of that kind); every row must have exactly `float_feature_count()` / `cat_feature_count()` entries. Return `InvalidArgumentError` with the exact mismatch — never let the C API segfault on ragged input.
- Marshal cat features to `const char***` (vector of vectors of `c_str()` pointers) without copying strings.
- `Predict` is `const` and must be **thread-safe** (CatBoost model application is read-only after load; document this and add a concurrent smoke test).
- `resultSize = docCount * dimension`.
- Map C-API `false` returns to `InternalError(GetErrorString())`.

---

## 3. L2 — TFS integration

### 3.1 Source adapter (StoragePath → Loader<cb::CatBoostModel>)

Template: `tensorflow_serving/servables/hashmap/hashmap_source_adapter.{h,cc}` (~60 lines). Use `SimpleLoaderSourceAdapter<StoragePath, cb::CatBoostModel>`:

- Loader fn: find the model file inside the version dir — accept exactly one `*.cbm` file (error if 0 or >1); call `cb::CatBoostModel::Load`.
- Resource estimate: file size × 1.5 via a custom estimator (fine for v1).
- Define `catboost_source_adapter.proto`:
  ```protobuf
  syntax = "proto3";
  package catboost.serving;
  message CatBoostSourceAdapterConfig {}   // registration key; empty for now
  ```
- Register: `REGISTER_STORAGE_PATH_SOURCE_ADAPTER(CatBoostSourceAdapterCreator, CatBoostSourceAdapterConfig);` and build the `cc_library` with **`alwayslink = 1`** (static-initializer registration — without alwayslink the linker drops it and the platform is silently missing).

### 3.2 gRPC API (own service — cleanest for categorical strings)

`catboost_predict.proto`:

```protobuf
syntax = "proto3";
package catboost.serving;
import "tensorflow_serving/apis/model.proto";  // reuse ModelSpec (name + version)

message Row {
  repeated float numeric = 1;      // order = training feature order
  repeated string categorical = 2; // order = training cat-feature order
}
message CatBoostPredictRequest {
  tensorflow.serving.ModelSpec model_spec = 1;
  repeated Row rows = 2;
  bool output_probabilities = 3;   // false => raw values
}
message CatBoostPredictResponse {
  tensorflow.serving.ModelSpec model_spec = 1;  // echoes resolved version
  int32 dimension = 2;
  repeated double values = 3;      // row-major [rows x dimension]
}
service CatBoostPredictionService {
  rpc Predict(CatBoostPredictRequest) returns (CatBoostPredictResponse);
}
```

`catboost_service_impl.cc` — mirror `model_servers/prediction_service_impl.cc`:

1. `server_core->GetServableHandle<cb::CatBoostModel>(request.model_spec(), &handle)` — this gives version resolution ("latest" if unset), model-not-found errors, and ref-counted safety across hot reloads **for free**.
2. Convert rows → the two feature matrices, call `Predict`/`PredictProba`, fill response, echo resolved version from `handle.id()`.
3. Map `absl::Status` → gRPC status.

### 3.3 Server binary

Copy `model_servers/main.cc` + `server.cc` into `tfs/main.cc`; register `CatBoostPredictionService` with the gRPC `ServerBuilder`; link the source adapter lib. Do **not** patch TFS core — everything is additive, keeping upstream rebases trivial.

Run config — `--platform_config_file`:

```
platform_configs {
  key: "catboost"
  value { source_adapter_config {
    [type.googleapis.com/catboost.serving.CatBoostSourceAdapterConfig] {}
  } }
}
```

and `--model_config_file` entries use `model_platform: "catboost"` with the standard layout `base_path/<version>/model.cbm`.

### 3.4 Bazel (L2)

Build **inside the TFS devel container** as a sibling package of the TFS workspace (simplest: `COPY`/mount `catboost_serving/` into `/tensorflow-serving/catboost_serving/` and add it to the workspace). Targets: `proto_library` + `cc_proto_library` + `cc_grpc_library` for the two protos (mirror `tensorflow_serving/apis/BUILD` patterns); `cc_library` for adapter (`alwayslink=1`) and service impl; `cc_binary` `catboost_model_server`. `third_party/catboost` here imports the **linux** `.so` only.

---

## 4. Test fixtures — real trained models (do this first)

`cb/testdata/train_test_models.py` (run once with `pip install catboost numpy`, on the Mac; commit outputs):

1. **`numeric_only.cbm`** — regression, 5 float features, synthetic `y = 3*x0 - 2*x1 + x2*x3 + noise`, 2000 rows, ~50 iterations, fixed `random_seed=42`.
2. **`categorical_only.cbm`** — binary classification, 3 cat features (e.g. `color∈{red,green,blue}`, `size∈{S,M,L,XL}`, `city∈{20 values}`), label depends on category combos, `cat_features=[0,1,2]`.
3. **`mixed_multiclass.cbm`** — **3-class** classification, 4 float + 2 cat features (`loss_function='MultiClass'`) — exercises `dimension()==3`.

The script also writes `expected_predictions.json`: for each model, **8 fixed input rows** (including: a category string never seen in training; extreme float values; for mixed — one row per class regime) and the Python-side outputs of both `predict(prediction_type='RawFormulaVal')` and `predict_proba`. These are the golden references — the C++ path must reproduce Python bit-for-close (rtol 1e-6; both call the same native evaluator, so agreement is tight).

---

## 5. THE unit test — `cb/catboost_model_test.cc` (GoogleTest, runs natively on the Mac)

```cpp
// Sketch — implement fully. Fixture loads expected_predictions.json (use nlohmann/json
// or a checked-in generated header) and the three .cbm files via bazel runfiles.

TEST(CatBoostModelTest, LoadReportsCorrectShapes) {
  auto m = cb::CatBoostModel::Load(TestData("mixed_multiclass.cbm")).value();
  EXPECT_EQ(m->float_feature_count(), 4);
  EXPECT_EQ(m->cat_feature_count(), 2);
  EXPECT_EQ(m->dimension(), 3);
}

TEST(CatBoostModelTest, NumericOnlyMatchesPython) {
  // Predict on the 8 golden rows with empty cat_features;
  // EXPECT_THAT(values, Pointwise(DoubleNear(1e-6), golden.raw));
}

TEST(CatBoostModelTest, CategoricalOnlyMatchesPython) { /* incl. unseen category row */ }

TEST(CatBoostModelTest, MixedMulticlassRawAndProbaMatchPython) {
  // raw: 8x3 values vs golden; proba: softmax vs Python predict_proba, and each
  // row of proba sums to 1 ± 1e-9.
}

TEST(CatBoostModelTest, BatchEqualsRowByRow) {
  // Predict(all 8 rows) == concat(Predict(row_i)) exactly.
}

TEST(CatBoostModelTest, InvalidInputsRejectedNotCrashing) {
  // wrong float count, wrong cat count, mismatched batch sizes, empty batch
  // -> InvalidArgumentError; missing file / garbage file -> NotFound/Internal.
}

TEST(CatBoostModelTest, ConcurrentPredictIsSafe) {
  // 8 threads x 200 predictions on one model instance; results all match golden.
}
```

Run: `bazel test //cb:catboost_model_test` — **must pass on macOS arm64 and on Linux in CI**.

### L2 tests (Docker only)

- `catboost_source_adapter_test.cc` — modeled on `hashmap_source_adapter_test.cc`: adapter converts a version dir into a Loader; loaded servable predicts golden row 0 correctly; dir with zero/two `.cbm` files → error.
- **End-to-end smoke** (shell/py test): start `catboost_model_server` with `mixed_multiclass` at version 1; gRPC `Predict` the 8 golden rows and compare; copy the model to `.../2/` and verify the response's echoed version flips to 2 without restart.

---

## 6. Build/execute order (do it in this sequence)

1. `train_test_models.py` → commit fixtures + golden JSON.
2. L1 `MODULE.bazel` + `third_party/catboost` (darwin arm64 dylib) → `catboost_model.{h,cc}` → make all §5 tests pass **natively on the Mac**.
3. Add Linux `.so` to the `select()`; verify L1 tests in a plain `ubuntu:24.04` container and in CI.
4. `docker/Dockerfile.devel` from `tensorflow/serving:latest-devel`; wire L2 Bazel packages.
5. Protos → source adapter (+ its test) → service impl → `main.cc`.
6. End-to-end smoke test in the container; document run command + example `grpcurl` call in `docs/`.

## 7. Known pitfalls (read before coding)

- **`alwayslink=1`** on the adapter lib, or the platform registration vanishes at link time.
- CatBoost C API takes `const char***` — pointer lifetimes must outlive the call; build the pointer arrays from stable `std::string` storage.
- `dimension()` for binary classification is 1 (single raw logit) — `PredictProba` must apply **sigmoid** when `dimension==1` and **softmax** when `>1`.
- Feature **order** is positional and must match training order; expose counts in errors so clients can debug. (v1 does not read feature names from the model; note as follow-up — the C API exposes feature names if wanted.)
- Bazel on the M4 under amd64 emulation: pin `--jobs`/`--local_ram_resources` or the build will OOM at 24 GB.
- Keep everything additive to the TFS tree — no patches to `tensorflow_serving/` itself.

## 8. Acceptance criteria

- [ ] All §5 L1 tests green on macOS arm64 and Linux x86_64.
- [ ] Adapter test green in Docker.
- [ ] E2E: serve all 3 model types via gRPC with predictions matching Python golden values (rtol 1e-6), incl. multiclass probabilities.
- [ ] Hot version swap works with no restart.
- [ ] `README` documents: training fixture regen, native L1 test loop on Mac, Docker L2 build, server run flags.
