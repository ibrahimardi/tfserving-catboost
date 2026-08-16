load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "catboost_model",
    srcs = ["catboost_model.cc"],
    hdrs = ["catboost_model.h"],
    deps = [
        "//third_party/catboost",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_test(
    name = "catboost_model_test",
    srcs = ["catboost_model_test.cc"],
    data = [
        "testdata/categorical_only.cbm",
        "testdata/expected_predictions.json",
        "testdata/mixed_multiclass.cbm",
        "testdata/numeric_only.cbm",
    ],
    deps = [
        ":catboost_model",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
