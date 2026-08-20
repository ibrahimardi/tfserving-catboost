load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Vendored from catboost v1.2.8 (Apache-2.0), must match the binaries pinned
# in MODULE.bazel: catboost/libs/model_interface/{c_api.h,model_calcer_wrapper.h}
CATBOOST_HEADERS = [
    "c_api.h",
    "model_calcer_wrapper.h",
]

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

cc_import(
    name = "libcatboostmodel",
    # Windows links through the import library; the DLL is the runtime piece.
    interface_library = select({
        "@platforms//os:windows": "@libcatboostmodel_windows_lib//file",
        "//conditions:default": None,
    }),
    shared_library = select({
        "@platforms//os:macos": "@libcatboostmodel_darwin//file",
        ":linux_x86_64": "@libcatboostmodel_linux_x86_64//file",
        ":linux_aarch64": "@libcatboostmodel_linux_aarch64//file",
        "@platforms//os:windows": "@libcatboostmodel_windows_dll//file",
    }),
)

cc_library(
    name = "catboost",
    hdrs = CATBOOST_HEADERS,
    deps = [":libcatboostmodel"],
)
