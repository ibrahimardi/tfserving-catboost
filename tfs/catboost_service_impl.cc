#include "tfs/catboost_service_impl.h"

#include <string>
#include <vector>

#include "cb/catboost_model.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"

namespace catboost {
namespace serving {

::grpc::Status CatBoostPredictionServiceImpl::Predict(
    ::grpc::ServerContext* /*context*/, const CatBoostPredictRequest* request,
    CatBoostPredictResponse* response) {
  // GetServableHandle gives version resolution ("latest" if unset),
  // model-not-found errors, and ref-counted safety across hot reloads.
  tensorflow::serving::ServableHandle<cb::CatBoostModel> handle;
  const absl::Status lookup_status =
      core_->GetServableHandle(request->model_spec(), &handle);
  if (!lookup_status.ok()) {
    return tensorflow::serving::ToGRPCStatus(lookup_status);
  }

  // Populate a feature block if the model uses it, or if a client sent values
  // for it anyway (so the wrapper can reject the mismatch with a clear error).
  std::vector<std::vector<float>> float_features;
  std::vector<std::vector<std::string>> cat_features;
  bool fill_float = handle->float_feature_count() > 0;
  bool fill_cat = handle->cat_feature_count() > 0;
  for (const Row& row : request->rows()) {
    if (row.numeric_size() > 0) fill_float = true;
    if (row.categorical_size() > 0) fill_cat = true;
  }
  for (const Row& row : request->rows()) {
    if (fill_float) {
      float_features.emplace_back(row.numeric().begin(), row.numeric().end());
    }
    if (fill_cat) {
      cat_features.emplace_back(row.categorical().begin(),
                                row.categorical().end());
    }
  }

  const absl::StatusOr<std::vector<double>> values =
      request->output_probabilities()
          ? handle->PredictProba(float_features, cat_features)
          : handle->Predict(float_features, cat_features);
  if (!values.ok()) {
    return tensorflow::serving::ToGRPCStatus(values.status());
  }

  response->set_dimension(handle->dimension());
  response->mutable_values()->Reserve(values->size());
  for (const double value : *values) response->add_values(value);
  auto* spec = response->mutable_model_spec();
  spec->set_name(request->model_spec().name());
  spec->mutable_version()->set_value(handle.id().version);
  return ::grpc::Status::OK;
}

}  // namespace serving
}  // namespace catboost
