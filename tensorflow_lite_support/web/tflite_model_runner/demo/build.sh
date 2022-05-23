#!/bin/bash

rm -rf dist/
mkdir dist/

# Build WASM related files.
bazel build //tensorflow_lite_support/web/tflite_model_runner/cc:tflite_model_runner_wasm
WASM_DIR=../../../../bazel-bin/tensorflow_lite_support/web/tflite_model_runner/cc/tflite_model_runner_wasm/
cp ${WASM_DIR}/tflite_model_runner_cc.js \
   ${WASM_DIR}/tflite_model_runner_cc.wasm \
   dist/

bazel build --copt="-msimd128" //tensorflow_lite_support/web/tflite_model_runner/cc:tflite_model_runner_wasm_simd
WASM_DIR=../../../../bazel-bin/tensorflow_lite_support/web/tflite_model_runner/cc/tflite_model_runner_wasm_simd/
cp ${WASM_DIR}/tflite_model_runner_cc_simd.js \
   ${WASM_DIR}/tflite_model_runner_cc_simd.wasm \
   dist/

bazel build --copt="-pthread" --copt="-msimd128" //tensorflow_lite_support/web/tflite_model_runner/cc:tflite_model_runner_wasm_threaded_simd
WASM_DIR=../../../../bazel-bin/tensorflow_lite_support/web/tflite_model_runner/cc/tflite_model_runner_wasm_threaded_simd/
cp ${WASM_DIR}/tflite_model_runner_cc_threaded_simd.js \
   ${WASM_DIR}/tflite_model_runner_cc_threaded_simd.wasm \
   ${WASM_DIR}/tflite_model_runner_cc_threaded_simd.worker.js \
   dist/


