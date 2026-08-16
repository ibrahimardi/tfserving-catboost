#include "cb/catboost_model.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

namespace cb {
namespace {

using ::testing::DoubleNear;
using ::testing::Pointwise;

constexpr double kRTol = 1e-6;

std::string TestData(const std::string& name) {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (srcdir != nullptr && workspace != nullptr) {
    return std::string(srcdir) + "/" + workspace + "/cb/testdata/" + name;
  }
  return "cb/testdata/" + name;
}

struct Golden {
  std::vector<std::vector<float>> float_features;
  std::vector<std::vector<std::string>> cat_features;
  std::vector<double> raw;    // row-major [rows x dimension]
  std::vector<double> proba;  // row-major [rows x n_classes]; empty for regression
  int raw_dimension = 0;
  int proba_classes = 0;
};

Golden LoadGolden(const std::string& model_name) {
  std::ifstream in(TestData("expected_predictions.json"));
  EXPECT_TRUE(in.good()) << "cannot open expected_predictions.json";
  nlohmann::json all = nlohmann::json::parse(in);
  const nlohmann::json& entry = all.at(model_name);

  Golden g;
  for (const auto& row : entry.at("float_features")) {
    g.float_features.push_back(row.get<std::vector<float>>());
  }
  for (const auto& row : entry.at("cat_features")) {
    g.cat_features.push_back(row.get<std::vector<std::string>>());
  }
  for (const auto& row : entry.at("raw")) {
    g.raw_dimension = row.size();
    for (double v : row) g.raw.push_back(v);
  }
  if (entry.contains("proba")) {
    for (const auto& row : entry.at("proba")) {
      g.proba_classes = row.size();
      for (double v : row) g.proba.push_back(v);
    }
  }
  return g;
}

std::unique_ptr<CatBoostModel> LoadModel(const std::string& file) {
  auto model = CatBoostModel::Load(TestData(file));
  EXPECT_TRUE(model.ok()) << model.status();
  return std::move(*model);
}

TEST(CatBoostModelTest, LoadReportsCorrectShapes) {
  auto mixed = LoadModel("mixed_multiclass.cbm");
  EXPECT_EQ(mixed->float_feature_count(), 4);
  EXPECT_EQ(mixed->cat_feature_count(), 2);
  EXPECT_EQ(mixed->dimension(), 3);

  auto numeric = LoadModel("numeric_only.cbm");
  EXPECT_EQ(numeric->float_feature_count(), 5);
  EXPECT_EQ(numeric->cat_feature_count(), 0);
  EXPECT_EQ(numeric->dimension(), 1);

  auto categorical = LoadModel("categorical_only.cbm");
  EXPECT_EQ(categorical->float_feature_count(), 0);
  EXPECT_EQ(categorical->cat_feature_count(), 3);
  EXPECT_EQ(categorical->dimension(), 1);
}

TEST(CatBoostModelTest, NumericOnlyMatchesPython) {
  auto model = LoadModel("numeric_only.cbm");
  Golden golden = LoadGolden("numeric_only");
  auto values = model->Predict(golden.float_features, {});
  ASSERT_TRUE(values.ok()) << values.status();
  EXPECT_THAT(*values, Pointwise(DoubleNear(kRTol), golden.raw));
}

TEST(CatBoostModelTest, CategoricalOnlyMatchesPython) {
  auto model = LoadModel("categorical_only.cbm");
  Golden golden = LoadGolden("categorical_only");
  // Row 5 holds categories never seen in training; CatBoost hashes raw
  // strings, so it must match Python exactly like the others.
  auto values = model->Predict({}, golden.cat_features);
  ASSERT_TRUE(values.ok()) << values.status();
  EXPECT_THAT(*values, Pointwise(DoubleNear(kRTol), golden.raw));

  // Binary model: dimension()==1, proba = sigmoid(raw) = P(class 1), which is
  // column 1 of Python's predict_proba.
  auto proba = model->PredictProba({}, golden.cat_features);
  ASSERT_TRUE(proba.ok()) << proba.status();
  ASSERT_EQ(golden.proba_classes, 2);
  std::vector<double> positive_class;
  for (size_t i = 1; i < golden.proba.size(); i += 2) {
    positive_class.push_back(golden.proba[i]);
  }
  EXPECT_THAT(*proba, Pointwise(DoubleNear(kRTol), positive_class));
}

TEST(CatBoostModelTest, MixedMulticlassRawAndProbaMatchPython) {
  auto model = LoadModel("mixed_multiclass.cbm");
  Golden golden = LoadGolden("mixed_multiclass");

  auto raw = model->Predict(golden.float_features, golden.cat_features);
  ASSERT_TRUE(raw.ok()) << raw.status();
  ASSERT_EQ(raw->size(), 8 * 3);
  EXPECT_THAT(*raw, Pointwise(DoubleNear(kRTol), golden.raw));

  auto proba = model->PredictProba(golden.float_features, golden.cat_features);
  ASSERT_TRUE(proba.ok()) << proba.status();
  EXPECT_THAT(*proba, Pointwise(DoubleNear(kRTol), golden.proba));
  for (size_t row = 0; row < proba->size(); row += 3) {
    double sum = (*proba)[row] + (*proba)[row + 1] + (*proba)[row + 2];
    EXPECT_NEAR(sum, 1.0, 1e-9);
  }
}

TEST(CatBoostModelTest, BatchEqualsRowByRow) {
  auto model = LoadModel("mixed_multiclass.cbm");
  Golden golden = LoadGolden("mixed_multiclass");

  auto batch = model->Predict(golden.float_features, golden.cat_features);
  ASSERT_TRUE(batch.ok()) << batch.status();
  std::vector<double> concatenated;
  for (size_t i = 0; i < golden.float_features.size(); ++i) {
    auto single = model->Predict({golden.float_features[i]},
                                 {golden.cat_features[i]});
    ASSERT_TRUE(single.ok()) << single.status();
    concatenated.insert(concatenated.end(), single->begin(), single->end());
  }
  EXPECT_EQ(*batch, concatenated);  // exact equality, same evaluator
}

TEST(CatBoostModelTest, InvalidInputsRejectedNotCrashing) {
  auto model = LoadModel("mixed_multiclass.cbm");

  // Wrong float count (expects 4).
  auto r = model->Predict({{1.0f, 2.0f}}, {{"circle", "wood"}});
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);

  // Wrong cat count (expects 2).
  r = model->Predict({{1, 2, 3, 4}}, {{"circle"}});
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);

  // Mismatched batch sizes.
  r = model->Predict({{1, 2, 3, 4}, {1, 2, 3, 4}}, {{"circle", "wood"}});
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);

  // Empty batch.
  r = model->Predict({}, {});
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);

  // Cat features passed to a numeric-only model.
  auto numeric = LoadModel("numeric_only.cbm");
  r = numeric->Predict({{1, 2, 3, 4, 5}}, {{"oops"}});
  EXPECT_EQ(r.status().code(), absl::StatusCode::kInvalidArgument);

  // Missing file -> NotFound; garbage file -> Internal.
  EXPECT_EQ(CatBoostModel::Load(TestData("does_not_exist.cbm")).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(CatBoostModel::Load(TestData("expected_predictions.json"))
                .status()
                .code(),
            absl::StatusCode::kInternal);
}

TEST(CatBoostModelTest, ConcurrentPredictIsSafe) {
  auto model = LoadModel("mixed_multiclass.cbm");
  Golden golden = LoadGolden("mixed_multiclass");

  auto reference = model->Predict(golden.float_features, golden.cat_features);
  ASSERT_TRUE(reference.ok()) << reference.status();

  constexpr int kThreads = 8;
  constexpr int kIterations = 200;
  std::vector<std::thread> threads;
  std::vector<bool> ok(kThreads, false);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kIterations; ++i) {
        auto values =
            model->Predict(golden.float_features, golden.cat_features);
        if (!values.ok() || *values != *reference) return;
      }
      ok[t] = true;
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_THAT(ok, ::testing::Each(true));
}

}  // namespace
}  // namespace cb
