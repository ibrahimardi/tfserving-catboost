# L2 BUILD file — consumed INSIDE the TensorFlow Serving tree, not by the L1
# module. docker/Dockerfile.devel copies this repo's cb/, third_party/ and
# tfs/ sources into /tensorflow-serving/tensorflow_serving/catboost_serving/
# and installs this file as that package's BUILD (paths below are relative to
# it). The catboost_serving package must live under //tensorflow_serving/...
# because most TFS deps are visibility-restricted to that subtree.
#
# `includes = ["."]` keeps the source include paths ("cb/...", "tfs/...",
# "third_party/...") identical between the L1 module and the TFS tree.

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

# ---------------------------------------------------------------- L1 sources

cc_import(
    name = "libcatboostmodel",
    shared_library = select({
        "@platforms//cpu:x86_64": "@libcatboostmodel_linux_x86_64//file",
        "@platforms//cpu:aarch64": "@libcatboostmodel_linux_aarch64//file",
    }),
)

cc_library(
    name = "catboost_c_api",
    hdrs = [
        "third_party/catboost/c_api.h",
        "third_party/catboost/model_calcer_wrapper.h",
    ],
    includes = ["."],
    deps = [":libcatboostmodel"],
)

cc_library(
    name = "catboost_model",
    srcs = ["cb/catboost_model.cc"],
    hdrs = ["cb/catboost_model.h"],
    includes = ["."],
    deps = [
        ":catboost_c_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# -------------------------------------------------------------------- protos

proto_library(
    name = "catboost_source_adapter_proto",
    srcs = ["tfs/proto/catboost_source_adapter.proto"],
)

cc_proto_library(
    name = "catboost_source_adapter_cc_proto",
    deps = [":catboost_source_adapter_proto"],
)

proto_library(
    name = "catboost_predict_proto",
    srcs = ["tfs/proto/catboost_predict.proto"],
    deps = ["//tensorflow_serving/apis:model_proto_pl"],
)

cc_proto_library(
    name = "catboost_predict_cc_proto",
    deps = [":catboost_predict_proto"],
)

cc_grpc_library(
    name = "catboost_predict_cc_grpc",
    srcs = [":catboost_predict_proto"],
    grpc_only = True,
    deps = [":catboost_predict_cc_proto"],
)

# ----------------------------------------------------------- TFS integration

cc_library(
    name = "catboost_source_adapter",
    srcs = ["tfs/catboost_source_adapter.cc"],
    hdrs = ["tfs/catboost_source_adapter.h"],
    includes = ["."],
    # Static-initializer registration of the "catboost" platform — without
    # alwayslink the linker drops it and the platform is silently missing.
    alwayslink = 1,
    deps = [
        ":catboost_model",
        ":catboost_source_adapter_cc_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/resources:resource_values",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "catboost_source_adapter_test",
    size = "medium",
    srcs = ["tfs/catboost_source_adapter_test.cc"],
    data = [
        "cb/testdata/mixed_multiclass.cbm",
        "cb/testdata/numeric_only.cbm",
    ],
    deps = [
        ":catboost_source_adapter",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "catboost_service_impl",
    srcs = ["tfs/catboost_service_impl.cc"],
    hdrs = ["tfs/catboost_service_impl.h"],
    includes = ["."],
    deps = [
        ":catboost_model",
        ":catboost_predict_cc_grpc",
        ":catboost_predict_cc_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:grpc_status_util",
        "//tensorflow_serving/model_servers:server_core",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_binary(
    name = "catboost_model_server",
    srcs = ["tfs/main.cc"],
    deps = [
        ":catboost_service_impl",
        ":catboost_source_adapter",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/config:platform_config_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/model_servers:model_service_impl",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/util:proto_util",
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
