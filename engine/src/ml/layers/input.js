import { Tensor } from "../math/tensor.js";
import { Layer } from "../layer.js";
import { InputType } from "../ml_types.js";
import { log, warn, error } from "../../utility/logging.js";

const number_name = "number";
const input_layer_error = "InputLayer not initialized correctly.";

/**
 * A simple ring buffer for training batches.
 *
 * Each training batch is an object with "input" and "target" properties.
 * This fixed‑capacity queue avoids array shifting overhead.
 */
export class TrainingQueue {
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
   * Returns a batch of samples from the queue.
   *
   * @param {number} batch_size - The size of the batch to return.
   * @returns {Object} An object with properties { input, target }.
   */
  #next_return_value = { input: null, target: null };
  next(batch_size, dispose_original = true) {
    const samples = [];

    this.#next_return_value.input = null;
    this.#next_return_value.target = null;

    // Dequeue samples up to batch_size
    while (samples.length < batch_size && this.length > 0) {
      const batch = this.shift();
      if (batch) {
        samples.push(batch);
      }
    }

    if (samples.length === 0) {
      return this.#next_return_value;
    }

    // Stack samples into batches
    // Note: Tensors are marked persistent=true when added, so they are safe here.
    // We make the *new* stacked tensors immediate=true as they are transient for this step.
    const input_batch = Tensor.stack(
      samples.map((s) => s.input),
      null,
      true /* immediate */
    );
    const target_batch = Tensor.stack(
      samples.map((s) => s.target),
      null,
      true /* immediate */
    );

    if (dispose_original) {
      // Dispose the original sample tensors now that they're batched
      for (let i = 0; i < samples.length; i++) {
        samples[i].input.dispose();
        samples[i].target.dispose();
      }
    }

    this.#next_return_value.input = input_batch;
    this.#next_return_value.target = target_batch;

    return this.#next_return_value;
  }

  /**
   * @returns {number} The current number of items in the queue.
   */
  get length() {
    return this.count;
  }
}

/**
 * Input layer class for handling input data and targets.
 *
 * This layer serves as the entry point for data into a neural network. It manages:
 * - A queue of training samples (input/target tensor pairs)
 * - Batching of samples for mini-batch training
 * - Proper tensor lifecycle management (persistence, disposal)
 *
 * The Input layer doesn't transform data like other layers - instead it:
 * 1. Stores incoming training data in its queue
 * 2. During forward pass, dequeues batches of samples
 * 3. Makes the target tensors available to loss layers
 * 4. Passes the input tensors to subsequent layers
 *
 * Configuration properties:
 * - capacity: Maximum number of samples that can be stored in the queue
 * - batch_size: Number of samples to combine into a single batch
 *
 * This layer is typically used as the root of a neural network subnet.
 *
 * Note: inputs can still be provided as arguments to the forward pass from external sources, and they will be used instead of the training
 * data in the queue, unless the input is null. If the target is null, the target from the effective context will be used.
 */
export class Input {
  /**
   * Initializes the InputLayer, creating the internal training queue.
   * Properties expected: { capacity: number, batch_size: number }
   */
  static initialize(layer) {
    const props = layer.properties;
    if (
      !props ||
      typeof props.capacity !== number_name ||
      typeof props.batch_size !== number_name
    ) {
      throw new Error(input_layer_error);
    }
    // Store the queue and batch size directly on the layer instance
    layer.training_queue = new TrainingQueue(props.capacity);
    layer.batch_size = props.batch_size;
    layer.current_input_batch = null; // To hold the latest batch for children
    layer.current_target_batch = null; // To hold the latest batch for loss functions
    layer.input_type = InputType.NUMERIC;
  }

  /**
   * Adds a single batch (input and target tensors) to the layer's queue.
   *
   * @param {LayerData} layer - The input layer instance.
   * @param {Tensor} input_batch - The input tensor for the batch.
   * @param {Tensor} target_batch - The target tensor for the batch.
   */
  static add_sample_batch(layer, input_batch, target_batch, mark_persistent = true) {
    if (!layer.training_queue) {
      error(input_layer_error);
      return;
    }
    // Ensure tensors passed in won't be disposed prematurely
    input_batch.persistent = mark_persistent;
    target_batch.persistent = mark_persistent;
    layer.training_queue.push({ input: input_batch, target: target_batch });
  }

  /**
   * Sets the input type for the input layer.
   *
   * @param {Layer} layer - The input layer instance.
   * @param {number} input_type - The input type to set.
   */
  static set_input_type(layer, input_type) {
    layer.input_type = input_type;
  }

  /**
   * Forward pass for the InputLayer.
   * Dequeues samples, creates a batch, stores target batch in context, returns input batch.
   */
  static forward(layer, input_tensor, target_tensor = null /* unused */) {
    if (!layer.training_queue) {
      return input_tensor;
    }

    let { input, target } = layer.training_queue.next(layer.batch_size);

    if (input === null) {
      input = input_tensor;
      target = target_tensor;
    }

    // Store batches for access by other layers/context
    layer.current_input_batch = input;
    layer.current_target_batch = target;

    // Store target batch in the effective training context for loss layers
    const context = Layer.get_effective_context(layer.id);
    context.target_batch = target;

    // Return the input batch to be passed to children
    return input;
  }

  /**
   * Backward pass for the InputLayer.
   * Input layers typically don't participate in backpropagation calculations, so this just forwards the gradient.
   */
  static backward(layer, grad_output_tensor, target_tensor = null /* unused */) {
    // Input layer is a source, gradient calculation stops here.
    // We just return the incoming gradient, signifying it's passed through unchanged
    // conceptually, though it won't actually be used by any parent.
    return grad_output_tensor;
  }
}
