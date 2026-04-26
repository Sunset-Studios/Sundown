import { TrainingQueue } from "./layers/input.js";
import { Layer } from "./layer.js";
import { MLOps } from "./ops/ops.js";
import { MLHopType } from "./ops/op_types.js";
import { NeuralArchitectureHelpers } from "./neural_architecture.js";
import { RandomAccessAllocator } from "../memory/allocator.js";

const function_name = "function";

class Subnet {
  id = null;
  subnet_id = null;
  train_step = null;
  infer = null;
  training_queue = null;
  cached_output = null;
}

/**
 * The MasterMind is an orchestrator that runs a continuous training loop in real time,
 * allowing subnets to be trained in small chunks while providing immediate access for inference.
 *
 * Subnets can be registered with:
 *    - a name,
 *    - an underlying subnet object,
 *    - a training step callback (which is expected to be:
 *          a) A time-based training step that processes a batch of data: function(deltaSeconds, inputTensor, targetTensor)
 *    - an inference callback.
 *

 * When no training step callback is provided but the subnet object has a .train() method,
 * the MasterMind assumes the subnet will be trained with a default training step that processes a batch of data: function(deltaSeconds, inputTensor, targetTensor)
 *
 * Similarly for inference, if no infer callback is provided and the subnet implements .predict(),
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
  static all_masterminds = [];

  /**
   * @param {Object} options - Configuration options.
   * @param {boolean} [options.enable_weight_sharing=false] - Whether to enable weight sharing.
   * @param {number} [options.weight_sharing_interval=1000] - The interval between weight sharing updates in milliseconds.
   * @param {number} [options.mini_batch_size=1] - The size of the mini-batch to use for training.
   */
  constructor(options = {}) {
    this.enable_weight_sharing = options.enable_weight_sharing || false;
    this.weight_sharing_interval = options.weight_sharing_interval || 1000; // in milliseconds
    this.weight_sharing_time_elapsed = 0.0;
    this.mini_batch_size = options.mini_batch_size || 1;
    this.subnets = new RandomAccessAllocator(256, Subnet); // subnet registry keyed by subnet ID.
    this.active_subnet = null; // ID of the subnet to receive training frames.
    this.paused = false;
    this.store = MLOps.new_op_store();
    this.store.register_observer(this);

    MasterMind.all_masterminds.push(this);
  }

  /**
   * Destroys the MasterMind instance.
   */
  destroy() {
    this.store.unregister_observer(this);
    MasterMind.all_masterminds = MasterMind.all_masterminds.filter((m) => m !== this);
  }

  on_ops_changed(store, op) {
    if (store !== this.store) return;

    switch (op.type) {
      case MLHopType.ADD_INPUT:
      case MLHopType.ADD_LAYER:
      case MLHopType.ADD_ACTIVATION:
      case MLHopType.ADD_LOSS:
      case MLHopType.CONNECT_LAYER:
      case MLHopType.DISCONNECT_LAYER:
      case MLHopType.DISCONNECT_LAYER_FROM_ALL:
      case MLHopType.RESET_MODEL:
        this.sync_registered_subnets();
        break;
    }
  }

  sync_registered_subnets() {
    const store_layer_ids = this.get_store_layer_ids();
    const root_ids = new Set(Layer.get_all_roots().filter((root_id) => store_layer_ids.has(root_id)));
    const active_subnet_entry = this.get_subnet(this.active_subnet);
    const active_subnet_root = active_subnet_entry?.subnet_id ?? null;

    for (const root_id of root_ids) {
      if (this.get_registered_subnet_id(root_id) === null) {
        this.register_subnet(root_id);
      }
    }

    for (let i = this.subnets.length - 1; i >= 0; i--) {
      const subnet = this.subnets.get(i);
      if (!root_ids.has(subnet.subnet_id) || !Layer.get(subnet.subnet_id)) {
        this.subnets.deallocate_at(i);

        if (this.active_subnet === subnet.id) {
          this.active_subnet = null;
        }
      }
    }

    this.refresh_subnet_ids();

    if (active_subnet_root !== null) {
      this.active_subnet = this.get_registered_subnet_id(active_subnet_root);
    }

    if (this.active_subnet === null) {
      if (this.subnets.length > 0) {
        this.active_subnet = 0;
      }
    }
  }

  get_store_layer_ids() {
    const layer_ids = new Set();

    for (let i = 0; i < this.store.hops.length; i++) {
      const hop = this.store.hops.get(i);
      switch (hop.type) {
        case MLHopType.ADD_INPUT:
        case MLHopType.ADD_LAYER:
        case MLHopType.ADD_ACTIVATION:
        case MLHopType.ADD_LOSS:
          if (Layer.get(hop.result)) {
            layer_ids.add(hop.result);
          }
          break;
      }
    }

    return layer_ids;
  }

  get_registered_subnet_id(subnet_id) {
    for (let i = 0; i < this.subnets.length; i++) {
      const subnet = this.subnets.get(i);
      if (subnet.subnet_id === subnet_id) {
        return subnet.id;
      }
    }

    return null;
  }

  refresh_subnet_ids() {
    for (let i = 0; i < this.subnets.length; i++) {
      this.subnets.get(i).id = i;
    }
  }

  /**
   * Registers a new subnet with the MasterMind.
   *
   * @param {string} subnet_id - Unique identifier for the subnet.
   * @param {Object} subnet_obj - The subnet object.
   * @param {number} [mini_batch_size=1] - The size of the mini-batch to use for training.
   * @param {function} [train_step_callback] - A training function. Depending on the model,
   * @param {function} [infer_callback] - A function(inputData) that returns inference results.
   * @returns {string} The registered model ID.
   */
  register_subnet(subnet_id, train_step_callback, infer_callback) {
    const existing_subnet_id = this.get_registered_subnet_id(subnet_id);
    if (existing_subnet_id !== null) {
      return existing_subnet_id;
    }

    // If no training callback is provided but the model object supports .train(),
    // wrap it so that it can be called with a batch.
    if (!train_step_callback) {
      train_step_callback = function (delta_time, input_tensor, target_tensor) {
        return NeuralArchitectureHelpers.train(subnet_id, input_tensor, target_tensor);
      };
    }

    // For inference, if no callback is supplied but the model has a .predict() method,
    // use it.
    if (!infer_callback) {
      infer_callback = function (input_data) {
        return NeuralArchitectureHelpers.predict(subnet_id, input_data);
      };
    }

    const subnet = this.subnets.allocate();
    const registered_subnet_id = this.subnets.length - 1;
    subnet.id = registered_subnet_id;
    subnet.subnet_id = subnet_id;
    subnet.train_step = train_step_callback;
    subnet.infer = infer_callback;
    subnet.training_queue = new TrainingQueue(8);
    subnet.cached_output = null;

    // If this is the first model registered, set it as the active training model.
    if (this.active_subnet === null) {
      this.active_subnet = registered_subnet_id;
    }

    return registered_subnet_id;
  }

  /**
   * Activates a model for continuous training.
   * @param {string} model_id - The model to activate.
   * @returns {boolean} True if successfully activated, false otherwise.
   */
  set_active_subnet(subnet_id) {
    if (this.get_subnet(subnet_id)) {
      this.active_subnet = subnet_id;
      return true;
    }
    return false;
  }

  /**
   * Returns a model by its ID.
   * @param {string} model_id - The ID of the model to retrieve.
   * @returns {Object} The model object.
   */
  get_subnet(subnet_id) {
    if (subnet_id === null || subnet_id < 0 || subnet_id >= this.subnets.length) {
      return null;
    }

    return this.subnets.get(subnet_id);
  }

  /**
   * Should be called on each frame.
   * @param {number} delta_time - Time elapsed since the last frame (in seconds).
   * @private
   */
  tick(delta_time) {
    if (this.paused) return;

    MLOps.reset();

    // If there are no models, do nothing.
    if (this.subnets.length === 0) {
      return;
    }

    // Update all models with their current batch in a round-robin manner.
    for (let i = 0; i < this.subnets.length; i++) {
      let subnet_entry = this.subnets.get(i);

      if (
        subnet_entry && typeof subnet_entry.train_step === function_name
      ) {
        const { input, target } = subnet_entry.training_queue.next(this.mini_batch_size);
        subnet_entry.train_step(delta_time, input, target);
      }
    }

    // If weight sharing is enabled and enough time has passed, perform live weight sharing

    this.weight_sharing_time_elapsed += delta_time;
    if (
      this.enable_weight_sharing &&
      this.weight_sharing_time_elapsed >= this.weight_sharing_interval
    ) {
      // For weight sharing, we consider only models that have a non-null cached_output.
      const candidate_subnets = [];
      for (let i = 0; i < this.subnets.length; i++) {
        const subnet = this.subnets.get(i);
        if (subnet.cached_output) {
          candidate_subnets.push(subnet);
        }
      }

      if (candidate_subnets.length >= 2) {
        // Select donor and receiver randomly (ensuring they are distinct).
        const donor_index = Math.floor(Math.random() * candidate_subnets.length);
        let receiver_index = Math.floor(Math.random() * candidate_subnets.length);
        while (receiver_index === donor_index) {
          receiver_index = Math.floor(Math.random() * candidate_subnets.length);
        }
        const donor = candidate_subnets[donor_index];
        const receiver = candidate_subnets[receiver_index];

        // Leak donor's cached output into receiver as new training data.
        // Here we assume that the donor's output can serve as both input and target.
        receiver.training_queue.push({
          input: donor.cached_output,
          target: donor.cached_output,
        });
      }

      this.weight_sharing_time_elapsed = 0.0;
    }

    MLOps.compile();
    MLOps.run();
  }

  /**
   * Adds a training batch for a model running in batch training mode.
   * @param {string} model_id - The model to which the training batch should be added.
   * @param {Object} input_tensor - Training inputs (e.g., { data: Float32Array, shape: [...] }).
   * @param {Object} target_tensor - Expected outputs (e.g., { data: Float32Array, shape: [...] }).
   * @returns {void}
   */
  add_training_batch(subnet_id, input_tensor, target_tensor) {
    const subnet_entry = this.get_subnet(subnet_id);
    if (!subnet_entry) {
      throw new Error(`Subnet "${subnet_id}" is not registered.`);
    }
    input_tensor.persistent = true;
    target_tensor.persistent = true;
    subnet_entry.training_queue.push({ input: input_tensor, target: target_tensor });
  }

  /**
   * Performs inference on the specified model.
   * @param {any} input_data - The input data for making predictions.
   * @param {string} [model_id] - Optional model ID; if omitted, the active model is used.
   * @returns {any} The inference results.
   */
  infer(input_data, subnet_id = null) {
    // Use the active model if none is explicitly provided.
    if (subnet_id === null) {
      subnet_id = this.active_subnet;
    }
    const subnet_entry = this.get_subnet(subnet_id);
    if (!subnet_entry || typeof subnet_entry.infer !== function_name) {
      throw new Error(`Subnet "${subnet_id}" is not registered or does not support inference.`);
    }
    return subnet_entry.infer(input_data);
  }

  /**
   * Optionally triggers a training step manually on a given model.
   * This processes the next batch if available.
   * @param {string} model_id - The model to train.
   * @param {number} delta_time - Duration of the training chunk.
   * @returns {void}
   */
  train_subnet_step(subnet_id, delta_time) {
    const subnet_entry = this.get_subnet(subnet_id);
    if (!subnet_entry) {
      throw new Error(`Subnet "${subnet_id}" is not registered.`);
    }
    if (
      typeof subnet_entry.train_step === function_name
    ) {
      const { input, target } = subnet_entry.training_queue.next(this.mini_batch_size);
      subnet_entry.train_step(delta_time, input, target);
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

  /**
   * Gets the stats for all models.
   * @returns {Object[]} An array of model stats.
   */
  get_subnet_stats() {
    const subnet_stats = [];

    for (let subnet_index = 0; subnet_index < this.subnets.length; subnet_index++) {
      const s = this.subnets.get(subnet_index);
      const context = Layer.get_effective_context(s.subnet_id);
      const name = context.name;
      const last_layer_ids = Layer.get_last_layer(s.subnet_id);

      let stats = [];
      for (let i = 0; i < last_layer_ids.length; i++) {
        const layer = Layer.get(last_layer_ids[i]);
        stats.push({
          name: `loss_${i}`,
          loss: layer.loss,
        });
      }

      subnet_stats.push({
        name: name,
        stats: stats,
      });
    }

    return subnet_stats;
  }

  /**
   * Creates a new MasterMind instance.
   * @param {Object} options - Configuration options.
   * @returns {MasterMind} A new MasterMind instance.
   */
  static create(options = {}) {
    const mastermind = new MasterMind(options);
    return mastermind;
  }
}
