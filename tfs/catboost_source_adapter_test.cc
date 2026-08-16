#include "tfs/catboost_source_adapter.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tfs/proto/catboost_source_adapter.pb.h"

namespace catboost {
namespace serving {
namespace {

using ::tensorflow::Env;
using ::tensorflow::serving::Loader;
using ::tensorflow::serving::ServableData;

std::string TestData(const std::string& name) {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  std::string prefix = "";
  if (srcdir != nullptr && workspace != nullptr) {
    prefix = std::string(srcdir) + "/" + workspace + "/";
  }
  return prefix + "tensorflow_serving/catboost_serving/cb/testdata/" + name;
}

// Creates a fresh version dir containing the given fixture files.
std::string MakeVersionDir(const std::string& dir_name,
                           const std::vector<std::string>& fixtures) {
  const std::string dir =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), dir_name);
  TF_CHECK_OK(Env::Default()->RecursivelyCreateDir(dir));
  for (const std::string& fixture : fixtures) {
    TF_CHECK_OK(Env::Default()->CopyFile(
        TestData(fixture), tensorflow::io::JoinPath(dir, fixture)));
  }
  return dir;
}

std::unique_ptr<CatBoostSourceAdapter> MakeAdapter() {
  return std::make_unique<CatBoostSourceAdapter>(
      CatBoostSourceAdapterConfig());
}

TEST(CatBoostSourceAdapterTest, VersionDirBecomesServableThatPredicts) {
  const std::string dir = MakeVersionDir("one_model", {"numeric_only.cbm"});
  ServableData<std::unique_ptr<Loader>> loader_data =
      MakeAdapter()->AdaptOneVersion({{"numeric", 1}, dir});
  TF_ASSERT_OK(loader_data.status());
  std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();

  TF_ASSERT_OK(loader->Load());
  const cb::CatBoostModel* model =
      loader->servable().get<cb::CatBoostModel>();
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->float_feature_count(), 5);
  EXPECT_EQ(model->dimension(), 1);

  // Golden row 0 of expected_predictions.json: all-zero features.
  auto values = model->Predict({{0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}, {});
  ASSERT_TRUE(values.ok()) << values.status();
  EXPECT_NEAR((*values)[0], 0.05377171207301851, 1e-6);

  loader->Unload();
}

TEST(CatBoostSourceAdapterTest, ResourceEstimateReflectsFileSize) {
  const std::string dir = MakeVersionDir("estimate", {"numeric_only.cbm"});
  ServableData<std::unique_ptr<Loader>> loader_data =
      MakeAdapter()->AdaptOneVersion({{"numeric", 1}, dir});
  TF_ASSERT_OK(loader_data.status());
  std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();

  tensorflow::uint64 file_size;
  TF_ASSERT_OK(Env::Default()->GetFileSize(
      tensorflow::io::JoinPath(dir, "numeric_only.cbm"), &file_size));
  tensorflow::serving::ResourceAllocation estimate;
  TF_ASSERT_OK(loader->EstimateResources(&estimate));
  ASSERT_EQ(estimate.resource_quantities_size(), 1);
  EXPECT_EQ(estimate.resource_quantities(0).quantity(),
            static_cast<tensorflow::uint64>(file_size * 1.5));
}

TEST(CatBoostSourceAdapterTest, EmptyDirIsError) {
  const std::string dir = MakeVersionDir("empty", {});
  ServableData<std::unique_ptr<Loader>> loader_data =
      MakeAdapter()->AdaptOneVersion({{"empty", 1}, dir});
  TF_ASSERT_OK(loader_data.status());
  std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();
  EXPECT_FALSE(loader->Load().ok());
}

TEST(CatBoostSourceAdapterTest, TwoCbmFilesIsError) {
  const std::string dir = MakeVersionDir(
      "two_models", {"numeric_only.cbm", "mixed_multiclass.cbm"});
  ServableData<std::unique_ptr<Loader>> loader_data =
      MakeAdapter()->AdaptOneVersion({{"two", 1}, dir});
  TF_ASSERT_OK(loader_data.status());
  std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();
  EXPECT_FALSE(loader->Load().ok());
}

}  // namespace
}  // namespace serving
}  // namespace catboost
