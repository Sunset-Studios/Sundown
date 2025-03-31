import { MLOps } from "../ops/ops.js";

/**
 * A tensor is a multi-dimensional array of floating-point numbers.
 * Tensors are used to store the weights, biases, and input/output data of a neural network.
 *
 * Tensors are managed by the Tensor class, which provides a pool of reusable tensors.
 * The Tensor class also provides utility methods for creating and manipulating tensors.
 */

export const TensorInitializer = {
  UNIFORM: 0,
  HE: 1,
  GLOROT: 2,
};

export class Tensor {
  // A pool of all allocated Tensor objects (by tensor ID/index).

  static pool = [];
  // A free list of tensor IDs that are available for reuse.
  static free_list = [];
  // Persistent flags stored per tensor, where index == tensor.id.
  static persistent_flags = [];
  // Used to assign new IDs when necessary.
  static next_tensor_id = 0;

  // Instance properties.
  id = null;
  shape = null;
  size = 0;
  data = null;
  array_type = null;

  /**
   * (Re)Initializes the Tensor instance.
   * Called by both the constructor and the create() factory when recycling.
   *
   * @param {Array|ArrayBuffer|TypedArray|null} data - initial data.
   * @param {number[]} shape - tensor shape.
   * @param {Function} ArrayType - Typed array constructor.
   */
  initialize(data, shape, batch_size = 1, ArrayType = Float32Array) {
    this.batch_size = batch_size;
    this.shape = shape;
    this.size = shape.reduce((a, b) => a * b, 1);
    this.array_type = ArrayType;

    if (data instanceof ArrayType) {
      this.data = data;
    } else if (data instanceof ArrayBuffer) {
      this.data = new ArrayType(data);
    } else if (Array.isArray(data)) {
      this.data = new ArrayType(data);
    } else {
      // Allocate a new typed array initialized to zeros.
      this.data = new ArrayType(this.size * this.batch_size);
    }

    // Only assign a new ID if this is a completely new instance.
    if (this.id === null) {
      this.id = Tensor.next_tensor_id++;
    }

    // Mark this tensor as non-persistent by default.
    Tensor.persistent_flags[this.id] = false;

    Tensor.debug_log_tensor(this.id, "initialize");
  }

  /**
   * Utility to create initialize a random tensor.
   */
  init_random(scale = 1) {
    MLOps.init_random(this, scale);
  }

  /**
   * Utility to initialize a random tensor with He initialization.
   */
  init_he() {
    MLOps.init_he(this);
  }

  /**
   * Utility to initialize a random tensor with Glorot initialization.
   */
  init_glorot(output_size) {
    MLOps.init_glorot(this, output_size);
  }

  /**
   * Utility to transpose a tensor.
   * @returns {Tensor} the transposed tensor.
   */
  transpose() {
    return MLOps.transpose(this);
  }

  /**
   * Utility to multiply two tensors.
   * @param {Tensor} tensor - the tensor to multiply.
   * @returns {Tensor} the product of the two tensors.
   */
  mat_mul(tensor) {
    return MLOps.mat_mul(this, tensor);
  }

  /**
   * Utility to add two tensors.
   * @param {Tensor} tensor - the tensor to add.
   * @returns {Tensor} the sum of the two tensors.
   */
  add(tensor) {
    return MLOps.add(this, tensor);
  }

  /**
   * Utility to subtract two tensors.
   * @param {Tensor} tensor - the tensor to subtract.
   * @returns {Tensor} the difference of the two tensors.
   */
  sub(tensor) {
    return MLOps.sub(this, tensor);
  }

  /**
   * Utility to subtract two tensors and assign the result to the first tensor.
   * @param {Tensor} tensor - the tensor to subtract.
   * @returns {Tensor} the difference of the two tensors.
   */
  sub_assign(tensor) {
    return MLOps.sub_assign(this, tensor);
  }

  /**
   * Utility to divide two tensors.
   * @param {Tensor} tensor - the tensor to divide.
   * @returns {Tensor} the quotient of the two tensors.
   */
  div(tensor) {
    return MLOps.div(this, tensor);
  }

  /**
   * Utility to scale a tensor.
   * @param {number} scale - the scale factor.
   * @returns {Tensor} the scaled tensor.
   */
  scale(scale) {
    return MLOps.scale(this, scale);
  }

  /**
   * Utility to apply the ReLU function to a tensor.
   * @returns {Tensor} the ReLU of the tensor.
   */
  relu() {
    return MLOps.relu(this);
  }

  /**
   * Utility to apply the ReLU function to a tensor.
   * @param {Tensor} grad - the gradient of the tensor.
   * @returns {Tensor} the ReLU of the tensor.
   */
  relu_backward(grad) {
    return MLOps.relu_backward(this, grad);
  }

  /**
   * Utility to apply the Tanh function to a tensor.
   * @returns {Tensor} the Tanh of the tensor.
   */
  tanh() {
    return MLOps.tanh(this);
  }

  /**
   * Utility to apply the Tanh function to a tensor.
   * @param {Tensor} grad - the gradient of the tensor.
   * @returns {Tensor} the Tanh of the tensor.
   */
  tanh_backward(grad) {
    return MLOps.tanh_backward(this, grad);
  }

  /**
   * Utility to apply the Sigmoid function to a tensor.
   * @returns {Tensor} the Sigmoid of the tensor.
   */
  sigmoid() {
    return MLOps.sigmoid(this);
  }

  /**
   * Utility to apply the Sigmoid function to a tensor.
   * @param {Tensor} grad - the gradient of the tensor.
   * @returns {Tensor} the Sigmoid of the tensor.
   */
  sigmoid_backward(grad) {
    return MLOps.sigmoid_backward(this, grad);
  }

  /**
   * Utility to apply the Softmax function to a tensor.
   * @returns {Tensor} the Softmax of the tensor.
   */
  softmax() {
    return MLOps.softmax(this);
  }

  /**
   * Utility to apply the Softmax function to a tensor.
   * @param {Tensor} grad - the gradient of the tensor.
   * @returns {Tensor} the Softmax of the tensor.
   */
  softmax_backward(grad) {
    return MLOps.softmax_backward(this, grad);
  }

  /**
   * Utility to fill a tensor with a specific value.
   *
   * @param {number} value - the value to fill the tensor with.
   * @param {number} offset - the offset to fill the tensor with.

   * @param {number} size - the size of the tensor to fill.
   * @returns {Tensor} the filled tensor.
   */
  fill(value, offset = 0, size = null) {
    return MLOps.fill(this, value, offset, size);
  }

  /**
   * Utility to extend a tensor.
   * @param {number[]} add_dims - the dimensions to add.
   * @param {number} fill_value - the value to fill the extended tensor with.
   * @returns {Tensor} the extended tensor.
   */
  extend(add_dims = [], fill_value = 0) {
    return MLOps.extend(this, add_dims, fill_value);
  }

  /**
   * Utility to reshape a tensor.
   * @param {number[]} shape - the shape to reshape the tensor to.
   * @returns {Tensor} the reshaped tensor.
   */
  reshape(shape) {
    return MLOps.reshape(this, shape);
  }

  /**
   * Utility to copy a tensor.
   * @param {Tensor} a - the tensor to copy.
   * @param {number} offset - the offset to copy the tensor from.
   * @param {number} size - the size of the tensor data to copy.
   * @returns {Tensor} the copied tensor.
   */
  copy(a, offset = 0, size = null) {
    return MLOps.copy(a, this, offset, size);
  }

  /**
   * Utility to clone a tensor.
   * @returns {Tensor} the cloned tensor.
   */
  clone() {
    return MLOps.clone(this);
  }

  /**
   * Utility to clip the L2 norm of a tensor in place.
   * @param {number} max_norm - the maximum L2 norm.
   */
  clip_l2_norm(max_norm = 1.0) {
    MLOps.clip_l2_norm(this, max_norm);
  }

  /**
   * Utility to reduce a tensor by its batch size.
   * @returns {Tensor} the reduced tensor.
   */
  batch_reduce_mean() {
    return MLOps.batch_reduce_mean(this);
  }

  /**
   * Utility to update the moment of a tensor in place using the Adam optimizer.
   * @param {Tensor} m_tensor - the first moment tensor.
   * @param {Tensor} v_tensor - the second moment tensor.
   * @param {Tensor} grad - the gradient tensor.
   * @param {number} beta1 - the first moment decay rate.
   * @param {number} beta2 - the second moment decay rate.
   * @param {number} t - the time step.
   * @param {number} epsilon - the epsilon value.
   * @param {number} learning_rate - the learning rate.
   */
  adam_moment_update(m_tensor, v_tensor, grad, beta1, beta2, t, epsilon, learning_rate) {
    MLOps.adam_moment_update(
      this,
      m_tensor,
      v_tensor,
      grad,
      beta1,
      beta2,
      t,
      epsilon,
      learning_rate
    );
  }

  /**
   * Dispose the tensor so that it becomes reusable.
   * This method should release resources (e.g. GPU buffers, if applicable) and
   * adds the tensor's ID to the free list.
   */
  dispose() {
    // (If there are GPU buffers or other resources, free them here.)
    Tensor.free_list.push(this.id);
    // Optionally, clear properties to help GC:
    this.data = null;
    this.shape = null;
    this.size = 0;
    this.batch_size = 0;
    this.array_type = null;
  }

  /**
   * Getter for the persistent flag.
   * The flag is stored in a static array for performance.
   * @returns {boolean} the persistent flag value.
   */
  get persistent() {
    return Tensor.persistent_flags[this.id];
  }

  /**
   * Setter for the persistent flag.
   * The flag is stored in a static array for performance.
   * @param {boolean} val - the new persistent flag value.
   */
  set persistent(val) {
    Tensor.persistent_flags[this.id] = val;
  }

  /**
   * Getter for the length of the tensor.
   * This is equivalent to the size of the tensor.
   * @returns {number} the size of the tensor.
   */
  get length() {
    return this.size;
  }

  /**
   * Getter for the batched length of the tensor.
   * This is equivalent to the size of the tensor multiplied by the batch size.
   * @returns {number} the batched length of the tensor.
   */
  get batched_length() {
    return this.size * this.batch_size;
  }

  /**
   * Factory method to create or recycle a Tensor.
   * This checks the free list for an available tensor and reinitializes it
   * instead of calling new every time.
   *
   * @param {Array|ArrayBuffer|TypedArray|null} data - initial data.
   * @param {number[]} shape - tensor shape.
   * @param {number} batch_size - the batch size of the tensor.
   * @param {Function} ArrayType - Typed array constructor.
   * @returns {Tensor} an allocated Tensor.
   */
  static create(data, shape, batch_size = 1, ArrayType = Float32Array) {
    if (Tensor.free_list.length > 0) {
      // Recycle an existing tensor.
      const tensor_id = Tensor.free_list.pop();
      const tensor = Tensor.pool[tensor_id];
      tensor.initialize(data, shape, batch_size, ArrayType);
      return tensor;
    } else {
      // No free tensor available: create a new one.
      const tensor = new Tensor();
      tensor.initialize(data, shape, batch_size, ArrayType);
      Tensor.pool[tensor.id] = tensor;
      return tensor;
    }
  }

  /**
   * Utility to create a zeros tensor.
   * Using Tensor.create(), this will automatically enable reuse.
   *
   * @param {number[]} shape - desired shape.
   * @param {number} batch_size - the batch size of the tensor.
   * @param {Function} ArrayType - Typed array constructor.
   * @returns {Tensor} a new zeros tensor.
   */
  static zeros(shape, batch_size = 1, ArrayType = Float32Array) {
    return Tensor.create(null, shape, batch_size, ArrayType);
  }

  /**
   * Utility to create a random tensor.
   * Using Tensor.create(), this will automatically enable reuse.
   *
   * @param {number[]} shape - desired shape.
   * @param {number} scale - scale factor for the random values.
   * @param {number} batch_size - the batch size of the tensor.
   * @param {Function} ArrayType - Typed array constructor.
   * @returns {Tensor} a new random tensor.
   */
  static random(shape, scale = 1, batch_size = 1, ArrayType = Float32Array) {
    const tensor = Tensor.create(null, shape, batch_size, ArrayType);
    tensor.init_random(scale);
    return tensor;
  }

  /**
   * Utility to create a random tensor with He initialization.
   * Using Tensor.create(), this will automatically enable reuse.
   *
   * @param {number[]} shape - desired shape.
   * @param {number} batch_size - the batch size of the tensor.
   * @param {Function} ArrayType - Typed array constructor.
   * @returns {Tensor} a new random tensor with He initialization.
   */
  static random_he(shape, batch_size = 1, ArrayType = Float32Array) {
    const tensor = Tensor.create(null, shape, batch_size, ArrayType);
    tensor.init_he();
    return tensor;
  }

  /**
   * Utility to create a random tensor with Glorot initialization.
   * Using Tensor.create(), this will automatically enable reuse.
   *
   * @param {number[]} shape - desired shape.
   * @param {number} output_size - the size of the output tensor.
   * @param {number} batch_size - the batch size of the tensor.
   * @param {Function} ArrayType - Typed array constructor.
   * @returns {Tensor} a new random tensor with Glorot initialization.
   */
  static random_glorot(shape, output_size, batch_size = 1, ArrayType = Float32Array) {
    const tensor = Tensor.create(null, shape, batch_size, ArrayType);
    tensor.init_glorot(output_size);
    return tensor;
  }

  /**
   * Utility to initialize a tensor.
   * @param {number[]} shape - desired shape.
   * @param {number} initializer - the initializer to use.
   * @param {number} output_size - the size of the output tensor.
   * @param {number} batch_size - the batch size of the tensor.
   * @param {Function} ArrayType - Typed array constructor.
   * @returns {Tensor} a new tensor initialized with the specified initializer.
   */
  static init_tensor(
    shape,
    initializer,
    output_size = null,
    batch_size = 1,
    ArrayType = Float32Array
  ) {
    if (initializer === TensorInitializer.UNIFORM) {
      return Tensor.random(shape, batch_size, ArrayType);
    } else if (initializer === TensorInitializer.HE) {
      return Tensor.random_he(shape, batch_size, ArrayType);
    } else if (initializer === TensorInitializer.GLOROT) {
      return Tensor.random_glorot(shape, output_size ?? shape[1], batch_size, ArrayType);
    }
  }

  /**
   * Utility to stack tensors.
   * This function takes an array of tensors and combines them into a single tensor
   * by stacking them along a new first dimension. The resulting tensor will have
   * a shape that reflects the number of tensors stacked and the shape of the
   * individual tensors. It is important that all tensors have the same shape
   * (excluding the first dimension) to ensure proper stacking.
   * @param {Tensor[]} tensors - the tensors to stack.
   * @param {Tensor} reference_tensor - the reference tensor to use for the shape.
   * @returns {Tensor} the stacked tensor.
   */
  static stack(tensors, reference_tensor = null, immediate = false) {
    if (tensors.length === 0) {
      throw new Error("Cannot stack an empty array of tensors.");
    }
    if (reference_tensor === null) {
      reference_tensor = tensors[0];
    }
    const result = Tensor.zeros(
      reference_tensor.shape,
      tensors.length * reference_tensor.batch_size
    );
    for (let i = 0; i < tensors.length; i++) {
      if (immediate) {
        result.data.set(tensors[i].data, i * reference_tensor.length * reference_tensor.batch_size);
      } else {
        result.copy(
          tensors[i],
          i * reference_tensor.length * reference_tensor.batch_size,
          tensors[i].length * tensors[i].batch_size
        );
      }
    }
    return result;
  }

  /**
   * Optionally, clean up any non-persistent tensors in the entire pool.

   * This walks through the pool and disposes tensors not marked persistent.
   */
  static cleanup() {
    for (let i = 0; i < Tensor.pool.length; i++) {
      if (Tensor.pool[i] && !Tensor.persistent_flags[i]) {
        Tensor.pool[i].dispose();
      }
    }
  }

  /**
   * Utility to set the debug tensor.
   * @param {number} id - the ID of the tensor to debug.
   */
  static set_debug_tensor(id) {
    Tensor.debug_tensor = id;
  }

  /**
   * Utility to clear the debug tensor.
   */
  static clear_debug_tensor() {
    Tensor.debug_tensor = null;
  }

  /**
   * Utility to log the tensor.
   * @param {number} id - the ID of the tensor to log.
   * @param {string} op_name - the name of the operation.
   */
  static debug_log_tensor(id, op_name) {
    if (Tensor.debug_tensor !== null && Tensor.debug_tensor === id && Tensor.pool[id]) {
      const tensor = Tensor.pool[id];
      console.log(
        `[Tensor ${tensor.id}]: (${op_name}) Data Length: ${tensor.data.length}, Batch Size: ${tensor.batch_size}, Data: ${tensor.data}, Data Shape: ${tensor.shape}`
      );
    }
  }
}
