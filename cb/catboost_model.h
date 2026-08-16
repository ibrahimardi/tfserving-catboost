#ifndef CATBOOST_SERVING_CB_CATBOOST_MODEL_H_
#define CATBOOST_SERVING_CB_CATBOOST_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace cb {

// RAII wrapper around the CatBoost C API (model_calcer_wrapper.h).
//
// Thread-safety: Predict/PredictProba are const and safe to call concurrently
// from multiple threads on the same instance — CatBoost model application is
// read-only after load.
class CatBoostModel {
 public:
  // Loads a .cbm model file. Returns NotFoundError if the file does not
  // exist, InternalError (with the CatBoost message) if it cannot be parsed.
  static absl::StatusOr<std::unique_ptr<CatBoostModel>> Load(
      const std::string& path);

  ~CatBoostModel();
  CatBoostModel(const CatBoostModel&) = delete;
  CatBoostModel& operator=(const CatBoostModel&) = delete;

  int float_feature_count() const { return float_feature_count_; }
  int cat_feature_count() const { return cat_feature_count_; }
  // Outputs per document: 1 for regression / binary raw value, K for K-class
  // multiclass.
  int dimension() const { return dimension_; }

  // Batch prediction of raw formula values.
  // float_features: [batch x float_feature_count()]; cat_features:
  // [batch x cat_feature_count()]. A block must be empty iff the model has 0
  // features of that kind; otherwise both non-empty blocks must have the same
  // batch size. Returns row-major [batch x dimension()] raw values.
  absl::StatusOr<std::vector<double>> Predict(
      const std::vector<std::vector<float>>& float_features,
      const std::vector<std::vector<std::string>>& cat_features) const;

  // Same inputs as Predict, but returns probabilities: sigmoid of the raw
  // value when dimension() == 1 (probability of the positive class, one value
  // per row), softmax over the K raw values when dimension() > 1.
  absl::StatusOr<std::vector<double>> PredictProba(
      const std::vector<std::vector<float>>& float_features,
      const std::vector<std::vector<std::string>>& cat_features) const;

 private:
  explicit CatBoostModel(void* handle);

  void* handle_;  // ModelCalcerHandle*
  int float_feature_count_ = 0;
  int cat_feature_count_ = 0;
  int dimension_ = 0;
};

}  // namespace cb

#endif  // CATBOOST_SERVING_CB_CATBOOST_MODEL_H_
