'use strict';

// built-in webnn delegate
importScripts('./tflite_model_runner_cc_simd.js');
importScripts('https://cdn.jsdelivr.net/npm/@webmachinelearning/webnn-polyfill/dist/webnn-polyfill.js');

var modelRunnerResult;
var modelRunner;
// Receive the message from the main thread
onmessage = async (message) => {
  if (message) {
    //Load model or infer depends on the first data
    switch (message.data.action) {
      case 'load':
        const startTs = performance.now();
        const modelPath = message.data.modelPath;
        // Load WASM module and model.
        const [module, modelArrayBuffer] = await Promise.all([
          tflite_model_runner_ModuleFactory(),
          (await fetch(modelPath)).arrayBuffer(),
        ]);
        // Load WASM module and model.
        const modelBytes = new Uint8Array(modelArrayBuffer);
        const offset = module._malloc(modelBytes.length);
        module.HEAPU8.set(modelBytes, offset);

        // Create model runner.
        modelRunnerResult =
          module.TFLiteWebModelRunner.CreateFromBufferAndOptions(
            offset,
            modelBytes.length,
            {
              numThreads: Math.min(
                4,
                Math.max(1, (navigator.hardwareConcurrency || 1) / 2)
              ),
              enableWebNNDelegate: message.data.enableWebNNDelegate,
              webNNDevicePreference: parseInt(message.data.webNNDevicePreference),
              webNNNumThreads: 0,
            }
          );

        if (!modelRunnerResult.ok()) {
          throw new Error(
            'Failed to create TFLiteWebModelRunner: ' +
              modelRunner.errorMessage()
          );
        }
        modelRunner = modelRunnerResult.value();
        const loadFinishedMs = (performance.now() - startTs).toFixed(2);
        postMessage(loadFinishedMs);
        break;

      case 'infer':

        //////////////////////////////////////////////////////////////////////////////
        // Get input and output info.

        const inputs = callAndDelete(modelRunner.GetInputs(), (results) =>
          convertCppVectorToArray(results)
        );
        const input = inputs[0];
        const outputs = callAndDelete(modelRunner.GetOutputs(), (results) =>
          convertCppVectorToArray(results)
        );
        const output = outputs[0];

        //////////////////////////////////////////////////////////////////////////////
        // Set input tensor data from the image (224 x 224 x 3).
        const inputBuffer = input.data();
        inputBuffer.set(message.data.buffer);

        //////////////////////////////////////////////////////////////////////////////
        // Infer, get output tensor, and sort by logit values in reverse.
        const start = performance.now();
        modelRunner.Infer();
        const inferTime = performance.now() - start;
        console.log(`Infer time: ${inferTime.toFixed(2)} ms`);

        let result = output.data();
        result = result.slice(0);
        postMessage({result, inferTime}, [result.buffer]);
      default:
        break;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Helper functions.

/** Converts the given c++ vector to a JS array. */
function convertCppVectorToArray(vector) {
  if (vector == null) return [];

  const result = [];
  for (let i = 0; i < vector.size(); i++) {
    const item = vector.get(i);
    result.push(item);
  }
  return result;
}

/**
 * Calls the given function with the given deletable argument, ensuring that
 * the argument gets deleted afterwards (even if the function throws an error).
 */
function callAndDelete(arg, func) {
  try {
    return func(arg);
  } finally {
    if (arg != null) arg.delete();
  }
}
