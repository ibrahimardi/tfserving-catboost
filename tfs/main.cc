// CatBoost model server: a lean TFS-based gRPC server that serves .cbm models
// with full TFS version management (hot reload of /models/<name>/<version>/).
//
// Run:
//   catboost_model_server \
//     --port=8500 \
//     --model_config_file=/config/models.conf \
//     --platform_config_file=/config/platform.conf
//
// The platform config must register the "catboost" platform:
//   platform_configs {
//     key: "catboost"
//     value { source_adapter_config {
//       [type.googleapis.com/catboost.serving.CatBoostSourceAdapterConfig] {}
//     } }
//   }
// and model configs use model_platform: "catboost".

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/config/model_server_config.pb.h"
#include "tensorflow_serving/config/platform_config.pb.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/model_servers/model_service_impl.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/util/proto_util.h"
#include "tfs/catboost_service_impl.h"

namespace {

using tensorflow::serving::AvailabilityPreservingPolicy;
using tensorflow::serving::ModelServerConfig;
using tensorflow::serving::ModelServiceImpl;
using tensorflow::serving::ParseProtoTextFile;
using tensorflow::serving::PlatformConfigMap;
using tensorflow::serving::ServerCore;

int RunServer(int port, const std::string& model_config_file,
              const std::string& platform_config_file,
              int file_system_poll_wait_seconds) {
  ServerCore::Options options;

  absl::Status status = ParseProtoTextFile<ModelServerConfig>(
      model_config_file, &options.model_server_config);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to parse --model_config_file: " << status.message();
    return 1;
  }
  status = ParseProtoTextFile<PlatformConfigMap>(platform_config_file,
                                                 &options.platform_config_map);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to parse --platform_config_file: "
               << status.message();
    return 1;
  }

  options.aspired_version_policy =
      std::make_unique<AvailabilityPreservingPolicy>();
  options.file_system_poll_wait_seconds = file_system_poll_wait_seconds;

  std::unique_ptr<ServerCore> core;
  status = ServerCore::Create(std::move(options), &core);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to create ServerCore: " << status.message();
    return 1;
  }

  ModelServiceImpl model_service(core.get());
  catboost::serving::CatBoostPredictionServiceImpl prediction_service(
      core.get());

  const std::string server_address = "0.0.0.0:" + std::to_string(port);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address,
                           ::grpc::InsecureServerCredentials());
  builder.RegisterService(&model_service);
  builder.RegisterService(&prediction_service);
  builder.SetMaxMessageSize(std::numeric_limits<tensorflow::int32>::max());
  std::unique_ptr<::grpc::Server> server = builder.BuildAndStart();
  if (server == nullptr) {
    LOG(ERROR) << "Failed to start gRPC server at " << server_address;
    return 1;
  }
  LOG(INFO) << "Running CatBoost gRPC ModelServer at " << server_address
            << " ...";
  server->Wait();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  tensorflow::int32 port = 8500;
  std::string model_config_file;
  std::string platform_config_file;
  tensorflow::int32 file_system_poll_wait_seconds = 1;

  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("port", &port, "TCP port to listen on for gRPC API"),
      tensorflow::Flag("model_config_file", &model_config_file,
                       "Path to a ModelServerConfig text proto; entries must "
                       "use model_platform: \"catboost\""),
      tensorflow::Flag("platform_config_file", &platform_config_file,
                       "Path to a PlatformConfigMap text proto registering "
                       "the catboost platform"),
      tensorflow::Flag("file_system_poll_wait_seconds",
                       &file_system_poll_wait_seconds,
                       "Interval in seconds between each poll of the "
                       "filesystem for new model versions (0 disables)"),
  };
  const std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list) ||
      model_config_file.empty() || platform_config_file.empty()) {
    std::cout << usage;
    return 1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  return RunServer(port, model_config_file, platform_config_file,
                   file_system_poll_wait_seconds);
}
