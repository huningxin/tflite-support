'use strict';

// tfjs-tflite + webnn external delegate
importScripts('https://cdn.jsdelivr.net/npm/@webmachinelearning/webnn-polyfill/dist/webnn-polyfill.js');
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu");
importScripts('./tfjs-tflite/tf-tflite.js');

let tfliteModel;

// Receive the message from the main thread
onmessage = async (message) => {
  if (message) {
    //Load model or infer depends on the first data
    switch (message.data.action) {
      case 'load':
        const modelPath = message.data.modelPath;
        const options = {
          numThreads: Math.min(
            4,
            Math.max(1, (navigator.hardwareConcurrency || 1) / 2)
          )
        };
        const enableWebNNDelegate = message.data.enableWebNNDelegate;
        const webNNDevicePreference = parseInt(message.data.webNNDevicePreference);

        if (enableWebNNDelegate) {
          options.delegatePath = './webnn_external_delegate_wasm.wasm';
        }

        tflite.setWasmPath('./tfjs-tflite/');
        const startTs = performance.now();
        // Load tflite model.
        tfliteModel = await tflite.loadTFLiteModel(modelPath, options);
        const loadFinishedMs = (performance.now() - startTs).toFixed(2);
        postMessage(loadFinishedMs);
        break;

      case 'infer':
        const numRuns = message.data.numRuns;
        let inputData = message.data.buffer;
        const inputTensor = tf.tensor(inputData, [1, 224, 224, 3]);
        const inferTimes = [];
        let outputTensor;
        for (let i = 0; i < numRuns; i++) {
          const start = performance.now();
          outputTensor = tfliteModel.predict(inputTensor);
          const inferTime = (performance.now() - start).toFixed(2);
          if (!outputTensor) return;
          console.log(`Infer time ${i + 1}: ${inferTime} ms`);
          inferTimes.push(Number(inferTime));
        }

        const result = outputTensor.arraySync()[0];
        result.shift(); // Remove the first logit which is the background noise.
        const sortedResult = result
          .map((logit, i) => {
            return { i, logit };
          })
          .sort((a, b) => b.logit - a.logit);
        postMessage({ sortedResult, inferTimes });
      default:
        break;
    }
  }
};
