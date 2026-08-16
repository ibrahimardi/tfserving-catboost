#ifndef CATBOOST_SERVING_TFS_CATBOOST_SERVICE_IMPL_H_
#define CATBOOST_SERVING_TFS_CATBOOST_SERVICE_IMPL_H_

#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tfs/proto/catboost_predict.grpc.pb.h"

namespace catboost {
namespace serving {

class CatBoostPredictionServiceImpl final
    : public CatBoostPredictionService::Service {
 public:
  explicit CatBoostPredictionServiceImpl(
      tensorflow::serving::ServerCore* core)
      : core_(core) {}

  ::grpc::Status Predict(::grpc::ServerContext* context,
                         const CatBoostPredictRequest* request,
                         CatBoostPredictResponse* response) override;

 private:
  tensorflow::serving::ServerCore* core_;
};

}  // namespace serving
}  // namespace catboost

#endif  // CATBOOST_SERVING_TFS_CATBOOST_SERVICE_IMPL_H_
