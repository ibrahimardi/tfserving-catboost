#include "tfs/catboost_source_adapter.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_serving/resources/resource_values.h"

namespace catboost {
namespace serving {
namespace {

using ::tensorflow::Env;
using ::tensorflow::serving::StoragePath;

// Finds the single *.cbm file inside the version directory 'dir'.
absl::Status FindCbmFile(const StoragePath& dir, std::string* cbm_path) {
  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(dir, &children));
  std::vector<std::string> cbm_files;
  for (const std::string& child : children) {
    if (absl::EndsWith(child, ".cbm")) cbm_files.push_back(child);
  }
  if (cbm_files.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "expected exactly one .cbm file in ", dir, ", found ",
        cbm_files.size(),
        cbm_files.empty() ? "" : ": " + absl::StrJoin(cbm_files, ", "));
  }
  *cbm_path = tensorflow::io::JoinPath(dir, cbm_files[0]);
  return absl::OkStatus();
}

absl::Status LoadCatBoostModel(const StoragePath& dir,
                               std::unique_ptr<cb::CatBoostModel>* model) {
  std::string cbm_path;
  TF_RETURN_IF_ERROR(FindCbmFile(dir, &cbm_path));
  absl::StatusOr<std::unique_ptr<cb::CatBoostModel>> loaded =
      cb::CatBoostModel::Load(cbm_path);
  if (!loaded.ok()) return loaded.status();
  *model = std::move(*loaded);
  return absl::OkStatus();
}

// Estimates the servable's RAM footprint as file size x 1.5.
absl::Status EstimateResources(
    const StoragePath& dir, tensorflow::serving::ResourceAllocation* estimate) {
  std::string cbm_path;
  TF_RETURN_IF_ERROR(FindCbmFile(dir, &cbm_path));
  tensorflow::uint64 file_size;
  TF_RETURN_IF_ERROR(Env::Default()->GetFileSize(cbm_path, &file_size));
  auto* ram = estimate->add_resource_quantities();
  ram->mutable_resource()->set_device(
      tensorflow::serving::device_types::kMain);
  ram->mutable_resource()->set_kind(tensorflow::serving::resource_kinds::kRamBytes);
  ram->set_quantity(static_cast<tensorflow::uint64>(file_size * 1.5));
  return absl::OkStatus();
}

}  // namespace

CatBoostSourceAdapter::CatBoostSourceAdapter(
    const CatBoostSourceAdapterConfig& /*config*/)
    : SimpleLoaderSourceAdapter<StoragePath, cb::CatBoostModel>(
          &LoadCatBoostModel, &EstimateResources) {}

CatBoostSourceAdapter::~CatBoostSourceAdapter() { Detach(); }

}  // namespace serving
}  // namespace catboost

// The registration macro references the registry types unqualified, so it
// must live inside namespace tensorflow::serving.
namespace tensorflow {
namespace serving {
namespace {

class CatBoostSourceAdapterCreator {
 public:
  static absl::Status Create(
      const catboost::serving::CatBoostSourceAdapterConfig& config,
      std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>*
          adapter) {
    adapter->reset(new catboost::serving::CatBoostSourceAdapter(config));
    return absl::OkStatus();
  }
};
REGISTER_STORAGE_PATH_SOURCE_ADAPTER(
    CatBoostSourceAdapterCreator,
    catboost::serving::CatBoostSourceAdapterConfig);

}  // namespace
}  // namespace serving
}  // namespace tensorflow
