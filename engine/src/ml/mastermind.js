import { MLOps } from "./ops.js";
import { Tensor } from "./tensor.js";

const function_name = "function";

/**
 * A simple ring buffer for training batches.

 *
 * Each training batch is an object with "input" and "target" properties.
 * This fixed‑capacity queue avoids array shifting overhead.
 */
class TrainingQueue {
  /**
   * @param {number} capacity - Maximum number of batches to hold.
   */
  constructor(capacity) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
    this.head = 0;
    this.tail = 0;
    this.count = 0;
  }

  /**
   * Adds a training batch to the queue.
   * If the queue is full, it overwrites the oldest batch.
   *
   * @param {Object} batch - An object with properties { input, target }.
   */
  push(batch) {
    if (this.count === this.capacity) {
      // Overwrite the oldest: advance head
      this.buffer[this.tail] = batch;
      this.tail = (this.tail + 1) % this.capacity;
      this.head = (this.head + 1) % this.capacity;
    } else {
      this.buffer[this.tail] = batch;
      this.tail = (this.tail + 1) % this.capacity;
      this.count++;
    }
  }

  /**
   * Removes and returns the oldest training batch.
   *
   * @returns {Object|null} The batch object or null if empty.
   */
  shift() {
    if (this.count === 0) return null;
    const batch = this.buffer[this.head];
    this.head = (this.head + 1) % this.capacity;
    this.count--;
    return batch;
  }

  /**
   * @returns {number} The current number of items in the queue.
   */
  get length() {
    return this.count;
  }
}

/**
 * The MasterMind is an orchestrator that runs a continuous training loop in real time,
 * allowing models to be trained in small chunks while providing immediate access for inference.
 *
 * Models can be registered with:
 *    - a name,
 *    - an underlying model object (e.g., an instance of NeuralModel or other ML models),
 *    - a training step callback (which is expected to be:
 *          a) A time-based training step that processes a batch of data: function(deltaSeconds, inputTensor, targetTensor)
 *    - an inference callback.
 *

 * When no training step callback is provided but the model object has a .train() method,
 * the MasterMind assumes the model will be trained with a default training step that processes a batch of data: function(deltaSeconds, inputTensor, targetTensor)
 *
 * Similarly for inference, if no infer callback is provided and the model implements .predict(),
 * that method will be used for inference.
 *
 * Additional features and ideas:
 *    - [Live weight sharing] The mastermind can leak layer outputs from random models into other models as new training
 *      data. This enables a live weight sharing / interpolation mechanism across models by default.
 *    - Adaptive scheduling based on performance metrics.
 *    - Event hooks for training/inference state changes.
 *    - Snapshotting of model states or real-time logging.
 */
export class MasterMind {
  static next_model_id = 0;

  /**
   * @param {Object} options - Configuration options.
   * @param {boolean} [options.enable_weight_sharing=false] - Whether to enable weight sharing.
   * @param {number} [options.weight_sharing_interval=1000] - The interval between weight sharing updates in milliseconds.
   */
  constructor(options = {}) {
    this.enable_weight_sharing = options.enable_weight_sharing || false;
    this.weight_sharing_interval = options.weight_sharing_interval || 1000; // in milliseconds
    this.weight_sharing_time_elapsed = 0.0;
    this.mini_batch_size = options.mini_batch_size || 1;
    this.models = {}; // model registry keyed by model ID.
    this.active_model = null; // ID of the model to receive training frames.
    this.paused = false;
  }

  /**
   * Registers a new model with the MasterMind.
   *
   * @param {string} model_id - Unique identifier for the model.
   * @param {Object} model_obj - The model object (e.g., NeuralModel instance).
   * @param {function} [train_step_callback] - A training function. Depending on the model,
   * @param {function} [infer_callback] - A function(inputData) that returns inference results.
   * @returns {string} The registered model ID.
   */
  register_model(model_name, model_obj, train_step_callback, infer_callback) {
    // If no training callback is provided but the model object supports .train(),
    // wrap it so that it can be called with a batch.
    if (!train_step_callback && typeof model_obj.train === function_name) {
      train_step_callback = function (delta_time, input_tensor, target_tensor) {
        return model_obj.train(input_tensor, target_tensor);
      };
    }

    // For inference, if no callback is supplied but the model has a .predict() method,
    // use it.
    if (!infer_callback && typeof model_obj.predict === function_name) {
      infer_callback = model_obj.predict.bind(model_obj);
    }

    const model_id = MasterMind.next_model_id++;

    this.models[model_id] = {
      id: model_id,
      name: model_name,
      model: model_obj,
      train_step: train_step_callback,
      infer: infer_callback,
      training_queue: new TrainingQueue(8),
      // Additional properties (e.g., for performance metrics) may be added here.
    };

    // If this is the first model registered, set it as the active training model.
    if (!this.active_model) {
      this.active_model = model_id;
    }

    return model_id;
  }

  /**
   * Activates a model for continuous training.
   * @param {string} model_id - The model to activate.
   * @returns {boolean} True if successfully activated, false otherwise.
   */
  set_active_model(model_id) {
    if (this.models[model_id]) {
      this.active_model = model_id;
      return true;
    }
    return false;
  }

  /**
   * Should be called on each frame.
   * @param {number} delta_time - Time elapsed since the last frame (in seconds).
   * @private
   */
  tick(delta_time) {
    if (this.paused) return;

    // If there are no models, do nothing.
    if (this.models.length === 0) {
      return;
    }

    // Update all models with their current batch in a round-robin manner.
    const model_ids = Object.keys(this.models);
    for (let i = 0; i < model_ids.length; i++) {
      const model_id = model_ids[i];
      let model_entry = this.models[model_id];

      if (model_entry.training_queue.length > 0 && typeof model_entry.train_step === function_name) {
        const samples = [];

        while (samples.length < this.mini_batch_size && model_entry.training_queue.length > 0) {
          const batch = model_entry.training_queue.shift();
          samples.push(batch);
        }

        const input_tensor_stack = Tensor.stack(samples.map((s) => s.input));
        const target_tensor_stack = Tensor.stack(samples.map((s) => s.target));

        for (let i = 0; i < samples.length; i++) {
          samples[i].input.dispose();
          samples[i].target.dispose();
        }

        model_entry.train_step(delta_time, input_tensor_stack, target_tensor_stack);
      }
    }

    // If weight sharing is enabled and enough time has passed, perform live weight sharing

    this.weight_sharing_time_elapsed += delta_time;
    if (
      this.enable_weight_sharing &&
      this.weight_sharing_time_elapsed >= this.weight_sharing_interval
    ) {
      // For weight sharing, we consider only models that have a non-null cached_output.
      const candidate_models = Object.values(this.models).filter(
        (m) => m.model.cached_output
      );

      if (candidate_models.length >= 2) {
        // Select donor and receiver randomly (ensuring they are distinct).
        const donor_index = Math.floor(Math.random() * candidate_models.length);
        let receiver_index = Math.floor(Math.random() * candidate_models.length);
        while (receiver_index === donor_index) {
          receiver_index = Math.floor(Math.random() * candidate_models.length);
        }
        const donor = candidate_models[donor_index];
        const receiver = candidate_models[receiver_index];

        // Leak donor's cached output into receiver as new training data.
        // Here we assume that the donor's output can serve as both input and target.
        receiver.training_queue.push({
          input: donor.model.cached_output,
          target: donor.model.cached_output,
        });
      }

      this.weight_sharing_time_elapsed = 0.0;
    }

    MLOps.compile();
    MLOps.run();
    MLOps.reset();
  }

  /**
   * Adds a training batch for a model running in batch training mode.
   * @param {string} model_id - The model to which the training batch should be added.
   * @param {Object} input_tensor - Training inputs (e.g., { data: Float32Array, shape: [...] }).
   * @param {Object} target_tensor - Expected outputs (e.g., { data: Float32Array, shape: [...] }).
   * @returns {void}
   */
  add_training_batch(model_id, input_tensor, target_tensor) {
    const model_entry = this.models[model_id];
    if (!model_entry) {
      throw new Error(`Model "${model_id}" is not registered.`);
    }
    input_tensor.persistent = true;
    target_tensor.persistent = true;
    model_entry.training_queue.push({ input: input_tensor, target: target_tensor });
  }

  /**
   * Performs inference on the specified model.
   * @param {any} input_data - The input data for making predictions.
   * @param {string} [model_id] - Optional model ID; if omitted, the active model is used.
   * @returns {any} The inference results.
   */
  infer(input_data, model_id = null) {
    // Use the active model if none is explicitly provided.
    if (!model_id) {
      model_id = this.active_model;
    }
    const model_entry = this.models[model_id];
    if (!model_entry || typeof model_entry.infer !== function_name) {
      throw new Error(`Model "${model_id}" is not registered or does not support inference.`);
    }
    return model_entry.infer(input_data);
  }

  /**
   * Optionally triggers a training step manually on a given model.
   * This processes the next batch if available.
   * @param {string} model_id - The model to train.
   * @param {number} delta_time - Duration of the training chunk.
   * @returns {void}
   */
  train_model_step(model_id, delta_time) {
    const model_entry = this.models[model_id];
    if (!model_entry) {
      throw new Error(`Model "${model_id}" is not registered.`);
    }
    if (model_entry.training_queue.length > 0 && typeof model_entry.train_step === function_name) {
      const batch = model_entry.training_queue.shift();
      model_entry.train_step(delta_time, batch.input, batch.target);
    }
  }

  /**
   * Pauses the automated training loop.
   */
  pause_training() {
    this.paused = true;
  }

  /**
   * Resumes the automated training loop.
   */
  resume_training() {
    this.paused = false;
  }
}
