#ifndef CATBOOST_SERVING_TFS_CATBOOST_SOURCE_ADAPTER_H_
#define CATBOOST_SERVING_TFS_CATBOOST_SOURCE_ADAPTER_H_

#include "cb/catboost_model.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"
#include "tfs/proto/catboost_source_adapter.pb.h"

namespace catboost {
namespace serving {

// A SourceAdapter that takes storage paths of model version directories
// (each containing exactly one *.cbm file) and produces loaders of
// cb::CatBoostModel servables.
class CatBoostSourceAdapter final
    : public tensorflow::serving::SimpleLoaderSourceAdapter<
          tensorflow::serving::StoragePath, cb::CatBoostModel> {
 public:
  explicit CatBoostSourceAdapter(const CatBoostSourceAdapterConfig& config);
  ~CatBoostSourceAdapter() override;

  CatBoostSourceAdapter(const CatBoostSourceAdapter&) = delete;
  CatBoostSourceAdapter& operator=(const CatBoostSourceAdapter&) = delete;
};

}  // namespace serving
}  // namespace catboost

#endif  // CATBOOST_SERVING_TFS_CATBOOST_SOURCE_ADAPTER_H_
