load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

cc_library(
    name = "tflite_model_runner",
    srcs = [
        "tflite_model_runner.cc",
        "tflite_model_runner.h",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/delegates/webnn:webnn_delegate",
    ],
)

# Threaded SIMD build
cc_binary(
    name = "tflite_model_runner_cc_threaded_simd",
    srcs = ["tflite_model_runner_wasm.cc"],
    copts = [
        "-fno-rtti",
        "-DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0",
    ],
    linkopts = [
        "--bind",
        "-Os",
        "-fno-rtti",
        "-s ASSERTIONS=0",
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s EXIT_RUNTIME=0",
        "-s MODULARIZE=1",
        "-s EXPORT_NAME=" + "'tflite_model_runner_ModuleFactory'",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        "-s DISABLE_EXCEPTION_CATCHING=1",
        # This cut 50k from glue js.
        "-s FILESYSTEM=0",
        "-s USE_PTHREADS=1",
        # Many x86-64 processors have 2 threads per core, so we divide by 2.
        "-s PTHREAD_POOL_SIZE=" +
        "'Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 1) / 2))'",
        "-s TOTAL_MEMORY=32MB",
        "-s ERROR_ON_UNDEFINED_SYMBOLS=0",
        "-s MALLOC=emmalloc",
        "-s INLINING_LIMIT=1",
        "-s USE_WEBNN=1",
    ],
    linkstatic = 1,
    deps = [
        ":tflite_model_runner",
        "//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow//tensorflow/lite:op_resolver",
        "@org_tensorflow//tensorflow/lite:tflite_with_xnnpack",  # link xnnpack
        "@org_tensorflow//tensorflow/lite:tflite_with_webnn",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

wasm_cc_binary(
    name = "tflite_model_runner_wasm_threaded_simd",
    cc_target = ":tflite_model_runner_cc_threaded_simd",
    simd = True,
    threads = "emscripten",
)

# SIMD build
cc_binary(
    name = "tflite_model_runner_cc_simd",
    srcs = ["tflite_model_runner_wasm.cc"],
    copts = [
        "-fno-rtti",
        "-DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0",
    ],
    linkopts = [
        "--bind",
        "-Os",
        "-fno-rtti",
        "-s ASSERTIONS=0",
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s EXIT_RUNTIME=0",
        "-s MODULARIZE=1",
        "-s EXPORT_NAME=" + "'tflite_model_runner_ModuleFactory'",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        "-s DISABLE_EXCEPTION_CATCHING=1",
        # This cut 50k from glue js.
        "-s FILESYSTEM=0",
        "-s TOTAL_MEMORY=32MB",
        "-s ERROR_ON_UNDEFINED_SYMBOLS=0",
        "-s MALLOC=emmalloc",
        "-s INLINING_LIMIT=1",
        "-s USE_WEBNN=1",
    ],
    linkstatic = 1,
    deps = [
        ":tflite_model_runner",
        "//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow//tensorflow/lite:op_resolver",
        "@org_tensorflow//tensorflow/lite:tflite_with_xnnpack",  # link xnnpack
        "@org_tensorflow//tensorflow/lite:tflite_with_webnn",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

wasm_cc_binary(
    name = "tflite_model_runner_wasm_simd",
    cc_target = ":tflite_model_runner_cc_simd",
    simd = True,
    threads = "off",
)

# Plain Wasm build
cc_binary(
    name = "tflite_model_runner_cc",
    srcs = ["tflite_model_runner_wasm.cc"],
    copts = [
        "-fno-rtti",
        "-DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0",
    ],
    linkopts = [
        "--bind",
        "-Os",
        "-fno-rtti",
        "-s ASSERTIONS=0",
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s EXIT_RUNTIME=0",
        "-s MODULARIZE=1",
        "-s EXPORT_NAME=" + "'tflite_model_runner_ModuleFactory'",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        "-s DISABLE_EXCEPTION_CATCHING=1",
        # This cut 50k from glue js.
        "-s FILESYSTEM=0",
        "-s TOTAL_MEMORY=32MB",
        "-s ERROR_ON_UNDEFINED_SYMBOLS=0",
        "-s MALLOC=emmalloc",
        "-s INLINING_LIMIT=1",
        "-s USE_WEBNN=1",
    ],
    linkstatic = 1,
    deps = [
        ":tflite_model_runner",
        "//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow//tensorflow/lite:op_resolver",
        "@org_tensorflow//tensorflow/lite:tflite_with_xnnpack",  # link xnnpack
        "@org_tensorflow//tensorflow/lite:tflite_with_webnn",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

wasm_cc_binary(
    name = "tflite_model_runner_wasm",
    cc_target = ":tflite_model_runner_cc",
    simd = False,
    threads = "off",
)
