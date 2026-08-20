#include "cb/catboost_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "third_party/catboost/model_calcer_wrapper.h"

namespace cb {
namespace {

absl::Status LastCatBoostError(const std::string& context) {
  return absl::InternalError(
      absl::StrFormat("%s: %s", context, GetErrorString()));
}

}  // namespace

CatBoostModel::CatBoostModel(void* handle) : handle_(handle) {
  float_feature_count_ = static_cast<int>(GetFloatFeaturesCount(handle_));
  cat_feature_count_ = static_cast<int>(GetCatFeaturesCount(handle_));
  dimension_ = static_cast<int>(GetDimensionsCount(handle_));
}

CatBoostModel::~CatBoostModel() { ModelCalcerDelete(handle_); }

absl::StatusOr<std::unique_ptr<CatBoostModel>> CatBoostModel::Load(
    const std::string& path) {
  // Portable existence check (std::filesystem is availability-gated on older
  // macOS deployment targets; POSIX stat doesn't exist under MSVC).
  if (!std::ifstream(path).good()) {
    return absl::NotFoundError(
        absl::StrFormat("model file not found: %s", path));
  }
  ModelCalcerHandle* handle = ModelCalcerCreate();
  if (handle == nullptr) {
    return absl::InternalError("ModelCalcerCreate failed");
  }
  if (!LoadFullModelFromFile(handle, path.c_str())) {
    absl::Status status =
        LastCatBoostError(absl::StrFormat("failed to load model %s", path));
    ModelCalcerDelete(handle);
    return status;
  }
  return std::unique_ptr<CatBoostModel>(new CatBoostModel(handle));
}

absl::StatusOr<std::vector<double>> CatBoostModel::Predict(
    const std::vector<std::vector<float>>& float_features,
    const std::vector<std::vector<std::string>>& cat_features) const {
  const bool wants_float = float_feature_count_ > 0;
  const bool wants_cat = cat_feature_count_ > 0;

  if (!wants_float && !float_features.empty()) {
    return absl::InvalidArgumentError(
        "model has no float features but float_features is non-empty");
  }
  if (!wants_cat && !cat_features.empty()) {
    return absl::InvalidArgumentError(
        "model has no categorical features but cat_features is non-empty");
  }
  if (wants_float && wants_cat &&
      float_features.size() != cat_features.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "batch size mismatch: %d float rows vs %d categorical rows",
        float_features.size(), cat_features.size()));
  }

  const size_t doc_count =
      wants_float ? float_features.size() : cat_features.size();
  if (doc_count == 0) {
    return absl::InvalidArgumentError("empty batch");
  }

  for (size_t i = 0; i < float_features.size(); ++i) {
    if (static_cast<int>(float_features[i].size()) != float_feature_count_) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "row %d has %d float features, model expects %d", i,
          float_features[i].size(), float_feature_count_));
    }
  }
  for (size_t i = 0; i < cat_features.size(); ++i) {
    if (static_cast<int>(cat_features[i].size()) != cat_feature_count_) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "row %d has %d categorical features, model expects %d", i,
          cat_features[i].size(), cat_feature_count_));
    }
  }

  // Pointer arrays reference the caller's stable std::string/std::vector
  // storage; nothing is copied.
  std::vector<const float*> float_rows;
  if (wants_float) {
    float_rows.reserve(doc_count);
    for (const auto& row : float_features) float_rows.push_back(row.data());
  }
  std::vector<std::vector<const char*>> cat_row_storage;
  std::vector<const char**> cat_rows;
  if (wants_cat) {
    cat_row_storage.reserve(doc_count);
    cat_rows.reserve(doc_count);
    for (const auto& row : cat_features) {
      std::vector<const char*> ptrs;
      ptrs.reserve(row.size());
      for (const auto& value : row) ptrs.push_back(value.c_str());
      cat_row_storage.push_back(std::move(ptrs));
      cat_rows.push_back(cat_row_storage.back().data());
    }
  }

  std::vector<double> result(doc_count * dimension_);
  if (!CalcModelPrediction(
          handle_, doc_count,
          wants_float ? float_rows.data() : nullptr,
          wants_float ? static_cast<size_t>(float_feature_count_) : 0,
          wants_cat ? cat_rows.data() : nullptr,
          wants_cat ? static_cast<size_t>(cat_feature_count_) : 0,
          result.data(), result.size())) {
    return LastCatBoostError("CalcModelPrediction failed");
  }
  return result;
}

absl::StatusOr<std::vector<double>> CatBoostModel::PredictProba(
    const std::vector<std::vector<float>>& float_features,
    const std::vector<std::vector<std::string>>& cat_features) const {
  absl::StatusOr<std::vector<double>> raw =
      Predict(float_features, cat_features);
  if (!raw.ok()) return raw.status();
  std::vector<double>& values = *raw;

  if (dimension_ == 1) {
    for (double& v : values) v = 1.0 / (1.0 + std::exp(-v));
    return values;
  }
  for (size_t row = 0; row < values.size(); row += dimension_) {
    const double max_value =
        *std::max_element(values.begin() + row,
                          values.begin() + row + dimension_);
    double sum = 0;
    for (int k = 0; k < dimension_; ++k) {
      values[row + k] = std::exp(values[row + k] - max_value);
      sum += values[row + k];
    }
    for (int k = 0; k < dimension_; ++k) values[row + k] /= sum;
  }
  return values;
}

}  // namespace cb
