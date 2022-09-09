'use strict';

importScripts('./tflite_model_runner_cc_simd.js');

var modelRunnerResult;
var modelRunner;
// Receive the message from the main thread
onmessage = async (message) => {
  if (message) {
    //Create delegate or infer depends on the first data  
    switch (message.data[0]) {
      case 'create':
        const startTs = Date.now();
        const modelPath = message.data[1];
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
              enableWebNNDelegate: message.data[2],
              webNNDevicePreference: parseInt(message.data[3]),
            }
          );

        if (!modelRunnerResult.ok()) {
          throw new Error(
            'Failed to create TFLiteWebModelRunner: ' +
              modelRunner.errorMessage()
          );
        }
        modelRunner = modelRunnerResult.value();
        const loadFinishedMs = Date.now() - startTs;
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

        const { vals, width, height } = message.data[1]
        if (!vals) return;
        const inputBuffer = input.data();
        const inputData = new Float32Array(inputBuffer.length);
        let pixelIndex = 0;
        for (let i = 0; i < width; i++) {
          for (let j = 0; j < height; j++) {
            const valStartIndex = pixelIndex * 4;
            const inputIndex = pixelIndex * 3;
            inputData[inputIndex] = (vals[valStartIndex] - 127.5) / 127.5;
            inputData[inputIndex + 1] =
              (vals[valStartIndex + 1] - 127.5) / 127.5;
            inputData[inputIndex + 2] =
              (vals[valStartIndex + 2] - 127.5) / 127.5;
            pixelIndex += 1;
          }
        }
        inputBuffer.set(inputData);

        //////////////////////////////////////////////////////////////////////////////
        // Infer, get output tensor, and sort by logit values in reverse.


        const numRuns = message.data[2]
        const inferTimes = [];
        for (let i = 0; i < numRuns; i++) {
          const start = performance.now();
          const success = modelRunner.Infer();
          const inferTime = (performance.now() - start).toFixed(2);
          if (!success) return;
          console.log(`Infer time ${i + 1}: ${inferTime} ms`);
          inferTimes.push(Number(inferTime));
        }

        const result = Array.from(output.data());
        result.shift(); // Remove the first logit which is the background noise.
        const sortedResult = result
          .map((logit, i) => {
            return { i, logit };
          })
          .sort((a, b) => b.logit - a.logit);
        postMessage({sortedResult, inferTimes})
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