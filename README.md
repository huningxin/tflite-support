This repo is a fork of the
[tflite-support](https://github.com/tensorflow/tflite-support) repo, with the
addition of the `tflite_model_runner` related code to show how it is built
using bazel and emsdk.

1. Add emscripten toolchain to the
   [WORKSPACE](https://github.com/jinjingforever/tflite-support/blob/master/WORKSPACE#L436-L449)
   file. This will set up emsdk and provide a useful build rule `wasm_cc_binary`
   that can be used to build WASM module from a cc_binary (see below).
2. I also added a small
   [patch](https://github.com/jinjingforever/tflite-support/blob/master/WORKSPACE#L169-L170)
   for the com_google_glog library. Without it, the emscripten build will fail
   on Mac/Linux. The glog
   [v0.5.0 release](https://github.com/google/glog/releases/tag/v0.5.0) has a
   fix for this issue, but it requires some TFLite code refactoring. I will talk
   to the TFLite team about this. For now we will just use the patch.
3. The cpp code is
   [here](https://github.com/jinjingforever/tflite-support/tree/master/tensorflow_lite_support/web/tflite_model_runner/cc).
   - `tflite_model_runner.*` implement the model runner.
   - `tflite_model_runner_wasm.cc` sets up the WASM related bindings.
   - The `BUILD` file uses the `wasm_cc_binary` rule to build the WASM module.
     It is currently linking with
     [XNNPACK](https://github.com/jinjingforever/tflite-support/blob/master/tensorflow_lite_support/web/tflite_model_runner/cc/BUILD#L57), with WASM
     SIMD and multi-threading
     [turned on](https://github.com/jinjingforever/tflite-support/blob/master/tensorflow_lite_support/web/tflite_model_runner/cc/BUILD#L65-L66).
4. The demo code is
   [here](https://github.com/jinjingforever/tflite-support/tree/master/tensorflow_lite_support/web/tflite_model_runner/demo). To run the demo:
   - Run `yarn && yarn build` in the demo directory. `yarn build` runs
     [build.sh](https://github.com/jinjingforever/tflite-support/blob/master/tensorflow_lite_support/web/tflite_model_runner/demo/build.sh)
     where you can see how the `bazel build` command is used and how to extract
     the output files.
   - Run `yarn start` to start the demo locally. It creates a model runner
     instance with the MobilenetV2 model and classfies an image. See the
     [script.js](https://github.com/jinjingforever/tflite-support/blob/master/tensorflow_lite_support/web/tflite_model_runner/demo/src/script.js) file
     for details about how things are used from the JS side.
   - Since this build configuration is for SIMD+multi-threading, please make
     sure Chrome has these two features
     [turned on](https://drive.google.com/file/d/13hqGwBOTJFoqMQVUAn_tiYp-uqQN_NUq/view?usp=sharing)
     from `chrome://flags`.
   - Run `yarn run electron` to start the demo in electron.js.
   - Run `yarn run electron-packager` to build an electron.js app of this demo.
5. The WebNN delegate support
   - `tensorflow_lite_webnn_delegate.patch`: the WebNN delegate implementation.
   - `emscripten_webnn_support.patch`: WebNN support of emscripten.
   - `emsdk.patch`: workaround the frozen cache setting of emscripten so WebNN support can be regenerated. As the cache is cleared, the first `yarn build` would fail. Please rerun, it would succeed.
   - Add `TFLiteWebModelRunnerOptions.enableWebNNDelegate` to enable or disable WebNN delegate.
   - Demo in browser is using webnn-polyfill.
   - Demo in electron.js could use [WebNN-native node.js binding](https://github.com/webmachinelearning/webnn-native/tree/main/node) for more details.
