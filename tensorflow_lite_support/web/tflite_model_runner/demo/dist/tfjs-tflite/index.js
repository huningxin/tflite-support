/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
export * from './tflite_model';
export * from './types/tflite_web_model_runner';
export * from './tflite_task_library_client/image_classifier';
export * from './tflite_task_library_client/image_segmenter';
export * from './tflite_task_library_client/object_detector';
export * from './tflite_task_library_client/nl_classifier';
export * from './tflite_task_library_client/bert_nl_classifier';
export * from './tflite_task_library_client/bert_qa';
export { setWasmPath } from './tflite_task_library_client/common';
export { getWasmFeatures } from './tflite_task_library_client/common';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5kZXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLXRmbGl0ZS9zcmMvaW5kZXgudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsY0FBYyxnQkFBZ0IsQ0FBQztBQUMvQixjQUFjLGlDQUFpQyxDQUFDO0FBQ2hELGNBQWMsK0NBQStDLENBQUM7QUFDOUQsY0FBYyw4Q0FBOEMsQ0FBQztBQUM3RCxjQUFjLDhDQUE4QyxDQUFDO0FBQzdELGNBQWMsNENBQTRDLENBQUM7QUFDM0QsY0FBYyxpREFBaUQsQ0FBQztBQUNoRSxjQUFjLHNDQUFzQyxDQUFDO0FBQ3JELE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxxQ0FBcUMsQ0FBQztBQUNoRSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0scUNBQXFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmV4cG9ydCAqIGZyb20gJy4vdGZsaXRlX21vZGVsJztcbmV4cG9ydCAqIGZyb20gJy4vdHlwZXMvdGZsaXRlX3dlYl9tb2RlbF9ydW5uZXInO1xuZXhwb3J0ICogZnJvbSAnLi90ZmxpdGVfdGFza19saWJyYXJ5X2NsaWVudC9pbWFnZV9jbGFzc2lmaWVyJztcbmV4cG9ydCAqIGZyb20gJy4vdGZsaXRlX3Rhc2tfbGlicmFyeV9jbGllbnQvaW1hZ2Vfc2VnbWVudGVyJztcbmV4cG9ydCAqIGZyb20gJy4vdGZsaXRlX3Rhc2tfbGlicmFyeV9jbGllbnQvb2JqZWN0X2RldGVjdG9yJztcbmV4cG9ydCAqIGZyb20gJy4vdGZsaXRlX3Rhc2tfbGlicmFyeV9jbGllbnQvbmxfY2xhc3NpZmllcic7XG5leHBvcnQgKiBmcm9tICcuL3RmbGl0ZV90YXNrX2xpYnJhcnlfY2xpZW50L2JlcnRfbmxfY2xhc3NpZmllcic7XG5leHBvcnQgKiBmcm9tICcuL3RmbGl0ZV90YXNrX2xpYnJhcnlfY2xpZW50L2JlcnRfcWEnO1xuZXhwb3J0IHtzZXRXYXNtUGF0aH0gZnJvbSAnLi90ZmxpdGVfdGFza19saWJyYXJ5X2NsaWVudC9jb21tb24nO1xuZXhwb3J0IHtnZXRXYXNtRmVhdHVyZXN9IGZyb20gJy4vdGZsaXRlX3Rhc2tfbGlicmFyeV9jbGllbnQvY29tbW9uJztcbiJdfQ==