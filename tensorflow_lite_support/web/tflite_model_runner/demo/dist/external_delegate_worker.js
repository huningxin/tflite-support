'use strict';

// tfjs-tflite + webnn external delegate
// importScripts('https://cdn.jsdelivr.net/npm/@webmachinelearning/webnn-polyfill/dist/webnn-polyfill.js');
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
          options.delegatePath = '../webnn_external_delegate_wasm.wasm';
        }

        tflite.setWasmPath('./tfjs-tflite/');
        const startTs = performance.now();
        // Load tflite model.
        tfliteModel = await tflite.loadTFLiteModel(modelPath, options);
        const loadFinishedMs = (performance.now() - startTs).toFixed(2);
        postMessage(loadFinishedMs);
        break;

      case 'infer':
        const inputData = message.data.buffer;
        const tensorStart = performance.now();
        const inputTensor = tf.tensor(inputData, [1, 224, 224, 3], 'float32');
        console.log('Time for tensor generatation: ', (performance.now() - tensorStart).toFixed(2));
        const start = performance.now();
        const outputTensor = tfliteModel.predict(inputTensor);
        const inferTime = performance.now() - start;
        console.log(`Infer time: ${inferTime.toFixed(2)} ms`);
        const start1 = performance.now()
        let result = outputTensor.dataSync();
        // Do copy since result's buffer is ArrayBuffer(33554432),
        // at index 0 is not detachable and could not be transferred.
        result = result.slice(0);
        console.log("time for output data sync and copy: ", (performance.now() - start1).toFixed());
        postMessage({result, inferTime}, [result.buffer]);
        tf.dispose(inputTensor);
        tf.dispose(outputTensor);
      default:
        break;
    }
  }
};
