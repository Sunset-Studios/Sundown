import { FrameAllocator } from "./memory/allocator.js";
import { Tensor } from "./tensor.js";
import { Name } from "./names.js";
import { logger } from "./logger.js";

// MLOps acts as a virtual machine for all ML operations.
// It is responsible for executing the operations on the appropriate backend.

// An enum of all supported low-level operations.
const MLOpType = Object.freeze({
  NONE: 0,
  INIT_RANDOM: 1,
  INIT_HE: 2,
  INIT_GLOROT: 3,
  MAT_MUL: 4,
  TRANSPOSE: 5,
  FILL: 6,
  EXTEND: 7,
  RESHAPE: 8,
  COPY: 9,
  CLONE: 10,
  ADD: 11,
  SUB: 12,
  SUB_ASSIGN: 13,
  DIV: 14,
  SCALE: 15,
  FUSED_MUL_ADD: 16,
  RELU: 17,
  RELU_BACKWARD: 18,
  TANH: 19,
  TANH_BACKWARD: 20,
  SIGMOID: 21,
  SIGMOID_BACKWARD: 22,
  MSE_LOSS: 23,
  MSE_LOSS_PRIME: 24,
  SOFTMAX: 25,
  SOFTMAX_BACKWARD: 26,
  CROSS_ENTROPY_LOSS: 27,
  CROSS_ENTROPY_LOSS_PRIME: 28,
  BINARY_CROSS_ENTROPY_LOSS: 29,
  BINARY_CROSS_ENTROPY_LOSS_PRIME: 30,
  CLIP_L2_NORM: 31,
  BATCH_REDUCE_MEAN: 32,
  ADAM_MOMENT_UPDATE: 33,
  WRITE_TENSOR: 34,
  READ_TENSOR: 35,
});


// A class that represents a low-level operation and all associated parameters.
class MLOp {
  type = MLOpType.NONE;
  param_start = 0;
  param_count = 0;
  result = -1;
}

// A class that represents the parameter list of a low-level operation.
class MLOpParams {
  data = null;
  #size = 0;
  #capacity = 0;

  constructor(size) {
    this.data = new Float64Array(size);
    this.#capacity = size;
  }

  reset() {
    this.#size = 0;
  }

  add(data, size = null) {
    let true_size;

    if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      true_size = size ?? data.length;
      if (this.#size + true_size > this.#capacity) {
        this.#capacity = this.#size + true_size * 2;
        const new_data = new Float64Array(this.#capacity);
        new_data.set(this.data);
        this.data = new_data;
      }
      this.data.set(data, this.#size);
      this.#size += true_size;
    } else {
      true_size = 1;
      if (this.#size + true_size > this.#capacity) {
        this.#capacity = this.#size + true_size * 2;
        const new_data = new Float64Array(this.#capacity);
        new_data.set(this.data);
        this.data = new_data;
      }
      this.data[this.#size] = data;
      this.#size += true_size;
    }
  }

  get(offset, size) {
    return this.data.subarray(offset, offset + size);
  }

  get_element(index) {
    return this.data[index];
  }

  set_element(index, value) {
    this.data[index] = value;
  }

  get length() {
    return this.#size;
  }
}

// A class that represents the CPU implementation of the low-level ML operations.
export class MLOpsCPU {
  ops = null;
  params = null;

  static init_random(a, scale = 1) {
    for (let i = 0; i < a.batched_length; i++) {
      a.data[i] = (Math.random() - 0.5) * scale;
    }
    if (isNaN(a.data[0])) {
      throw new Error("NaN in init_random(): tensor " + a.id);
    }
    Tensor.debug_log_tensor(a.id, "init_random");
  }

  static init_he(a) {
    for (let i = 0; i < a.batched_length; i++) {
      a.data[i] = (Math.random() - 0.5) * Math.sqrt(2 / a.length);
    }
    if (isNaN(a.data[0])) {
      throw new Error("NaN in init_he(): tensor " + a.id);
    }
    Tensor.debug_log_tensor(a.id, "init_he");
  }

  static init_glorot(a, output_size) {
    const fan_in = a.length;
    const fan_out = output_size;
    const std = Math.sqrt(6 / (fan_in + fan_out));
    for (let i = 0; i < a.batched_length; i++) {
      a.data[i] = (Math.random() - 0.5) * std;
    }
    if (isNaN(a.data[0])) {
      throw new Error("NaN in init_glorot(): tensor " + a.id);
    }
    Tensor.debug_log_tensor(a.id, "init_glorot");
  }

  static mat_mul(result, a, b) {
    if (b.batch_size != 1 && a.batch_size !== b.batch_size) {
      throw new Error(
        `Batch size mismatch in mat_mul(): a.batch_size=${a.batch_size}, b.batch_size=${b.batch_size}`
      );
    }
    const [m, n] = a.shape;
    const [o, p] = b.shape;

    if (n !== o) {
      throw new Error(`Incompatible shapes for matrix multiplication: ${a.shape} and ${b.shape}`);
    }
    const b_batch_size = b.batch_size;
    const batch_size = a.batch_size;
    for (let i = 0; i < batch_size; i++) {
      const a_batch_offset = i * m * n;
      const b_batch_offset = b_batch_size > 1 ? i * n * p : 0;
      const result_batch_offset = i * m * p;
      for (let j = 0; j < m; j++) {
        for (let l = 0; l < p; l++) {
          let sum = 0;
          for (let k = 0; k < n; k++) {
            sum += a.data[a_batch_offset + j * n + k] * b.data[b_batch_offset + k * p + l];
          }
          result.data[result_batch_offset + j * p + l] = sum;
        }
      }
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in mat_mul(): tensor " + result.id + ", a: " + a.id + ", b: " + b.id);
    }
    Tensor.debug_log_tensor(result.id, "mat_mul");
  }

  static fill(result, value, offset = 0, size = null) {
    const end = size === null ? result.batched_length : offset + size;
    for (let i = offset; i < end; i++) {
      result.data[i] = value;
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in fill(): tensor " +
          result.id +
          ", value: " +
          value +
          ", offset: " +
          offset +
          ", size: " +
          size
      );
    }
    Tensor.debug_log_tensor(result.id, "fill");
  }

  static extend(result, a, add_dims = [], fill_value = 0) {
    // Ensure batch sizes match.
    if (a.batch_size !== result.batch_size) {
      throw new Error(
        `Batch size mismatch in extend(): a.batch_size=${a.batch_size}, result.batch_size=${result.batch_size}`
      );
    }
    const dims = a.shape.length;
    // Validate that the result shape (per sample) matches a.shape + add_dims elementwise.
    for (let d = 0; d < dims; d++) {
      const expected = a.shape[d] + (add_dims[d] || 0);
      if (result.shape[d] !== expected) {
        throw new Error(
          `Shape mismatch on dimension ${d}: expected ${expected}, got ${result.shape[d]}`
        );
      }
    }
    // Pre-calculate strides for result and for a for one sample, assuming row-major order.
    const result_strides = new Array(dims);
    const a_strides = new Array(dims);
    result_strides[dims - 1] = 1;
    a_strides[dims - 1] = 1;
    for (let d = dims - 2; d >= 0; --d) {
      result_strides[d] = result_strides[d + 1] * result.shape[d + 1];
      a_strides[d] = a_strides[d + 1] * a.shape[d + 1];
    }
    // Compute the number of elements in one sample.
    const a_sample_length = a.length;
    const result_sample_length = result.length;
    // Loop over each batch.
    for (let b = 0; b < a.batch_size; b++) {
      // Offsets for the current batch (assumed to be stored consecutively).
      const a_offset = b * a_sample_length;
      const result_offset = b * result_sample_length;
      // Iterate over every element in the result sample.
      for (let i = 0; i < result_sample_length; i++) {
        let r = i;
        let a_index = 0;
        let in_bounds = true;
        // Compute the multi-index for the current element using the precomputed strides.
        for (let d = 0; d < dims; d++) {
          const idx = Math.floor(r / result_strides[d]);
          r %= result_strides[d];
          // If the index is outside the bounds of the original sample, mark as out-of-bound.
          if (idx >= a.shape[d]) {
            in_bounds = false;
            break;
          }
          a_index += idx * a_strides[d];
        }
        // Write from the original tensor or set to the fill_value.
        result.data[result_offset + i] = in_bounds ? a.data[a_offset + a_index] : fill_value;
      }
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in extend(): tensor " +
          result.id +
          ", a: " +
          a.id +
          ", add_dims: " +
          add_dims +
          ", fill_value: " +
          fill_value
      );
    }
    Tensor.debug_log_tensor(result.id, "extend");
  }

  static reshape(result, a, shape) {
    // Update the result tensor properties.
    result.shape = shape;
    result.size = shape.reduce((acc, curr) => acc * curr, 1);
    // Create a new data array for the reshaped tensor.
    // Here, result.batch_size gives the number of samples (batches).
    const total_elements = result.size * result.batch_size;
    const new_data = new a.array_type(total_elements);
    // The number of elements per sample in the original tensor.
    // (Assuming a.length represents the per-sample element count.)
    const original_sample_count = a.length;
    // Determine how many elements we can copy (if new shape is smaller than the existing sample, we copy as much as we can).
    const copy_count = Math.min(original_sample_count, result.size);
    // Process each batch separately.
    for (let b = 0; b < result.batch_size; b++) {
      // Compute the starting offset for the current batch in both arrays.
      const src_offset = b * original_sample_count;
      const dst_offset = b * result.size;
      // Copy the shared amount of data.
      new_data.set(a.data.subarray(src_offset, src_offset + copy_count), dst_offset);
      // If the result sample has more elements than the original sample,
      // fill the remaining elements with zeros.
      if (result.size > original_sample_count) {
        new_data.fill(0, dst_offset + copy_count, dst_offset + result.size);
      }
    }
    result.data = new_data;
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in reshape(): tensor " + result.id + ", a: " + a.id + ", shape: " + shape
      );
    }
    Tensor.debug_log_tensor(result.id, "reshape");
  }

  static copy(a, result, offset = 0, size = null) {
    const end = size === null ? result.batched_length : offset + size;
    for (let i = offset; i < end; i++) {
      result.data[i] = a.data[i];
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in copy(): tensor " +
          result.id +
          ", a: " +
          a.id +
          ", offset: " +
          offset +
          ", size: " +
          size
      );
    }
    Tensor.debug_log_tensor(result.id, "copy");
  }

  static clone(result, a) {
    result.data.set(a.data);
    if (result.data.some(isNaN)) {
      throw new Error("NaN in clone(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "clone");
  }

  static transpose(result, a) {
    for (let b = 0; b < a.batch_size; b++) {
      const batch_stride = b * a.shape[0] * a.shape[1];
      for (let i = 0; i < a.shape[0]; i++) {
        for (let j = 0; j < a.shape[1]; j++) {
          result.data[batch_stride + j * a.shape[0] + i] =
            a.data[batch_stride + i * a.shape[1] + j];
        }
      }
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in transpose(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "transpose");
  }

  static add(result, a, b) {
    if (a.batch_size !== b.batch_size) throw new Error("Batch size mismatch in add().");
    if (a.length !== b.length) throw new Error("Length mismatch in add().");
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = a.data[i] + b.data[i];
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in add(): tensor " + result.id + ", a: " + a.id + ", b: " + b.id);
    }
    Tensor.debug_log_tensor(result.id, "add");
  }

  static sub(result, a, b) {
    if (a.batch_size !== b.batch_size) throw new Error("Batch size mismatch in sub().");
    if (a.length !== b.length) throw new Error("Length mismatch in sub().");
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = a.data[i] - b.data[i];
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in sub(): tensor " + result.id + ", a: " + a.id + ", b: " + b.id);
    }
    Tensor.debug_log_tensor(result.id, "sub");
  }

  static sub_assign(result, b) {
    if (result.batch_size !== b.batch_size) throw new Error("Batch size mismatch in sub_assign().");
    if (result.length !== b.length) throw new Error("Length mismatch in sub_assign().");
    for (let i = 0; i < result.batched_length; i++) {
      result.data[i] -= b.data[i];
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in sub_assign(): tensor " + result.id + ", b: " + b.id);
    }
    Tensor.debug_log_tensor(result.id, "sub_assign");
  }

  static div(result, a, b) {
    if (a.batch_size !== b.batch_size) throw new Error("Batch size mismatch in div().");
    if (a.length !== b.length) throw new Error("Length mismatch in div().");
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = a.data[i] / b.data[i];
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in div(): tensor " + result.id + ", a: " + a.id + ", b: " + b.id);
    }
    Tensor.debug_log_tensor(result.id, "div");
  }

  static dot(result, a, b) {
    if (a.batch_size !== b.batch_size) throw new Error("Batch size mismatch in dot().");
    if (a.length !== b.length) throw new Error("Length mismatch in dot().");
    for (let b = 0; b < a.batch_size; b++) {
      const a_batch_offset = b * a.length;
      const b_batch_offset = b * b.length;
      for (let i = 0; i < a.length; i++) {
        result.data[b] += a.data[a_batch_offset + i] * b.data[b_batch_offset + i];
      }
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in dot(): tensor " + result.id + ", a: " + a.id + ", b: " + b.id);
    }
    Tensor.debug_log_tensor(result.id, "dot");
  }

  static scale(result, a, b) {
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = a.data[i] * b;
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in scale(): tensor " + result.id + ", a: " + a.id + ", b: " + b);
    }
    Tensor.debug_log_tensor(result.id, "scale");
  }

  static relu(result, a) {
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = Math.max(0, a.data[i]);
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in relu(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "relu");
  }

  static relu_backward(result, a, grad) {
    if (a.batch_size !== grad.batch_size)
      throw new Error("Batch size mismatch in relu_backward().");
    if (a.length !== grad.length) throw new Error("Length mismatch in relu_backward().");
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = a.data[i] > 0 ? grad.data[i] : 0;
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in relu_backward(): tensor " + result.id + ", a: " + a.id + ", grad: " + grad.id
      );
    }
    Tensor.debug_log_tensor(result.id, "relu_backward");
  }

  static tanh(result, a) {
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = Math.tanh(a.data[i]);
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in tanh(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "tanh");
  }

  static tanh_backward(result, a, grad) {
    if (a.batch_size !== grad.batch_size)
      throw new Error("Batch size mismatch in tanh_backward().");
    if (a.length !== grad.length) throw new Error("Length mismatch in tanh_backward().");
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = grad.data[i] * (1 - a.data[i] * a.data[i]);
    }

    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in tanh_backward(): tensor " + result.id + ", a: " + a.id + ", grad: " + grad.id
      );
    }
    Tensor.debug_log_tensor(result.id, "tanh_backward");
  }

  static sigmoid(result, a) {
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = 1 / (1 + Math.exp(-a.data[i]));
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in sigmoid(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "sigmoid");
  }

  static sigmoid_backward(result, a, grad) {
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = grad.data[i] * a.data[i] * (1 - a.data[i]);
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in sigmoid_backward(): tensor " + result.id + ", a: " + a.id + ", grad: " + grad.id);
    }
    Tensor.debug_log_tensor(result.id, "sigmoid_backward");
  }

  static mse_loss(result, target, output, enabled_logging = false, name = null) {
    if (target.batch_size !== output.batch_size)
      throw new Error("Batch size mismatch in mse_loss().");
    if (target.length !== output.length) throw new Error("Length mismatch in mse_loss().");
    let loss = 0;
    for (let i = 0; i < target.batched_length; i++) {
      const diff = target.data[i] - output.data[i];
      loss += diff * diff;
    }
    if (target.batched_length > 0) {
      result.data[0] = loss / target.batched_length;
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in mse_loss(): tensor " +
          result.id +
          ", target: " +
          target.id +
          ", output: " +
          output.id
      );
    }
    if (enabled_logging) {
      logger.log(`${Name.string(name)} MSE Loss: ${result.data[0]}`);
    }
    Tensor.debug_log_tensor(result.id, "mse_loss");
  }

  static mse_loss_prime(result, target, output) {
    if (target.batch_size !== output.batch_size)
      throw new Error("Batch size mismatch in mse_loss_prime().");
    if (target.length !== output.length) throw new Error("Length mismatch in mse_loss_prime().");
    for (let i = 0; i < target.batched_length; i++) {
      result.data[i] = (2 * (output.data[i] - target.data[i])) / target.length;
    }

    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in mse_loss_prime(): tensor " +
          result.id +
          ", target: " +
          target.id +
          ", output: " +
          output.id
      );
    }
    Tensor.debug_log_tensor(result.id, "mse_loss_prime");
  }

  static softmax(result, a) {
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = Math.exp(a.data[i]) / a.data.length;
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in softmax(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "softmax");
  }

  static softmax_backward(result, a, grad) {
    if (a.batch_size !== grad.batch_size)
      throw new Error("Batch size mismatch in softmax_backward().");
    if (a.length !== grad.length) throw new Error("Length mismatch in softmax_backward().");
    for (let i = 0; i < a.batched_length; i++) {
      result.data[i] = a.data[i] * (1 - a.data[i]) * grad.data[i];
    }

    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in softmax_backward(): tensor " + result.id + ", a: " + a.id + ", grad: " + grad.id
      );
    }
    Tensor.debug_log_tensor(result.id, "softmax_backward");
  }

  static cross_entropy_loss(result, target, output, enabled_logging = false, name = null) {
    if (target.batch_size !== output.batch_size)
      throw new Error("Batch size mismatch in cross_entropy_loss().");
    if (target.length !== output.length)
      throw new Error("Length mismatch in cross_entropy_loss().");
    let loss = 0;
    for (let i = 0; i < target.batched_length; i++) {
      loss -= target.data[i] * Math.log(output.data[i]);
    }
    result.data[0] = loss / target.length;
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in cross_entropy_loss(): tensor " +
          result.id +
          ", target: " +
          target.id +
          ", output: " +
          output.id
      );
    }
    if (enabled_logging) {
      logger.log(`${Name.string(name)} Cross Entropy Loss: ${result.data[0]}`);
    }
    Tensor.debug_log_tensor(result.id, "cross_entropy_loss");
  }

  static cross_entropy_loss_prime(result, target, output) {
    if (target.batch_size !== output.batch_size)
      throw new Error("Batch size mismatch in cross_entropy_loss_prime().");
    if (target.length !== output.length)
      throw new Error("Length mismatch in cross_entropy_loss_prime().");
    for (let i = 0; i < target.batched_length; i++) {
      result.data[i] = -target.data[i] / output.data[i];
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in cross_entropy_loss_prime(): tensor " +
          result.id +
          ", target: " +
          target.id +
          ", output: " +
          output.id
      );
    }
    Tensor.debug_log_tensor(result.id, "cross_entropy_loss_prime");
  }

  static binary_cross_entropy_loss(result, target, output, enabled_logging = false, name = null) {
    if (target.batch_size !== output.batch_size)
      throw new Error("Batch size mismatch in binary_cross_entropy_loss().");
    if (target.length !== output.length)
      throw new Error("Length mismatch in binary_cross_entropy_loss().");
    // Define a small epsilon to prevent log(0) or log of negative values.
    const EPS = 1e-7;
    let loss = 0;
    for (let i = 0; i < target.batched_length; i++) {
      // Clamp output.data[i] so that it is within [EPS, 1 - EPS].
      const pred = Math.min(Math.max(output.data[i], EPS), 1 - EPS);
      
      // Calculate the loss using the clamped prediction.
      loss -= target.data[i] * Math.log(pred) +
              (1 - target.data[i]) * Math.log(1 - pred);
    }
    result.data[0] = loss / target.length;
    if (result.data.some(isNaN)) {
      throw new Error("NaN in binary_cross_entropy_loss(): tensor " + result.id +
                      ", target: " + target.id + ", output: " + output.id);
    }
    if (enabled_logging) {
      logger.log(`${Name.string(name)} Binary Cross Entropy Loss: ${result.data[0]}`);
    }
    Tensor.debug_log_tensor(result.id, "binary_cross_entropy_loss");
  }

  static binary_cross_entropy_loss_prime(result, target, output) {
    if (target.batch_size !== output.batch_size)
      throw new Error("Batch size mismatch in binary_cross_entropy_loss_prime().");
    if (target.length !== output.length)
      throw new Error("Length mismatch in binary_cross_entropy_loss_prime().");
    // Define a small epsilon to ensure predictions are within a valid range.
    const EPS = 1e-7;
    for (let i = 0; i < target.batched_length; i++) {
      // Clamp the output to [EPS, 1 - EPS] to avoid division by zero.
      const pred = Math.min(Math.max(output.data[i], EPS), 1 - EPS);
      
      result.data[i] = -target.data[i] / pred + (1 - target.data[i]) / (1 - pred);
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in binary_cross_entropy_loss_prime(): tensor " + result.id + ", target: " + target.id + ", output: " + output.id);
    }
    Tensor.debug_log_tensor(result.id, "binary_cross_entropy_loss_prime");
  }

  static clip_l2_norm(result, max_norm) {
    const l2_norm = Math.sqrt(result.data.reduce((sum, value) => sum + value * value, 0));
    const scale_factor = Math.min(1, max_norm / l2_norm);
    for (let i = 0; i < result.batched_length; i++) {
      result.data[i] *= scale_factor;
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in clip_l2_norm(): tensor " + result.id + ", max_norm: " + max_norm);
    }
    Tensor.debug_log_tensor(result.id, "clip_l2_norm");
  }

  static batch_reduce_mean(result, a) {
    for (let j = 0; j < a.length; j++) {
      let sum = 0;
      for (let i = 0; i < a.batch_size; i++) {
        sum += a.data[i * a.length + j];
      }
      result.data[j] = sum / a.batch_size;
    }
    if (result.data.some(isNaN)) {
      throw new Error("NaN in batch_reduce_mean(): tensor " + result.id + ", a: " + a.id);
    }
    Tensor.debug_log_tensor(result.id, "batch_reduce_mean");
  }

  static adam_moment_update(
    result,
    m_tensor,
    v_tensor,
    grad,
    beta1,
    beta2,
    t,
    epsilon,
    learning_rate
  ) {
    // Optional: Early check for NaN or infinite values in the gradient.
    if (grad.data.some((value) => !Number.isFinite(value))) {
      throw new Error("Invalid gradient values detected before Adam update.");
    }
    // Loop over each element in the tensor data.
    for (let i = 0; i < result.batched_length; i++) {
      // Update biased first moment estimate.
      m_tensor.data[i] = beta1 * m_tensor.data[i] + (1 - beta1) * grad.data[i];
      // Update biased second raw moment estimate.
      v_tensor.data[i] = beta2 * v_tensor.data[i] + (1 - beta2) * (grad.data[i] * grad.data[i]);
      // Compute bias-corrected first moment.
      const m_hat = m_tensor.data[i] / (1 - Math.pow(beta1, t));
      // Compute bias-corrected second moment.
      const v_hat = v_tensor.data[i] / (1 - Math.pow(beta2, t));
      // Update the variable (in-place).
      result.data[i] -= (learning_rate * m_hat) / (Math.sqrt(v_hat) + epsilon);
    }
    if (result.data.some(isNaN)) {
      throw new Error(
        "NaN in adam_moment_update(): tensor " +
          result.id +
          ", m_tensor: " +
          m_tensor.id +
          ", v_tensor: " +
          v_tensor.id +
          ", grad: " +
          grad.id +
          ", beta1: " +
          beta1 +
          ", beta2: " +
          beta2 +
          ", t: " +
          t +
          ", epsilon: " +
          epsilon +
          ", learning_rate: " +
          learning_rate
      );
    }
    Tensor.debug_log_tensor(result.id, "adam_moment_update");
  }

  static compile(ops, params) {
    this.ops = ops;
    this.params = params;
  }

  static run() {
    const tensors = Tensor.pool;

    for (let i = 0; i < this.ops.length; i++) {
      const op = this.ops.get(i);
      const params_start = op.param_start;

      switch (op.type) {
        case MLOpType.INIT_RANDOM:
          this.init_random(tensors[op.result], this.params.get_element(params_start));
          break;
        case MLOpType.INIT_HE:
          this.init_he(tensors[op.result]);
          break;
        case MLOpType.INIT_GLOROT:
          this.init_glorot(tensors[op.result], this.params.get_element(params_start));
          break;
        case MLOpType.MAT_MUL:
          this.mat_mul(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.FILL:
          this.fill(
            tensors[op.result],
            this.params.get_element(params_start),
            this.params.get_element(params_start + 1),
            this.params.get_element(params_start + 2)
          );
          break;
        case MLOpType.EXTEND:
          const add_dims = this.params.get(params_start + 1, op.param_count - 1);
          this.extend(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            add_dims,
            this.params.get_element(params_start + op.param_count - 1)
          );
          break;
        case MLOpType.RESHAPE:
          this.reshape(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            this.params.get(params_start + 1, op.param_count - 1)
          );
          break;
        case MLOpType.COPY:
          this.copy(
            tensors[this.params.get_element(params_start)],
            tensors[op.result],
            this.params.get_element(params_start + 1),
            this.params.get_element(params_start + 2)
          );
          break;
        case MLOpType.CLONE:
          this.clone(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;
        case MLOpType.TRANSPOSE:
          this.transpose(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;

        case MLOpType.ADD:
          this.add(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.SUB:
          this.sub(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.SUB_ASSIGN:
          this.sub_assign(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;
        case MLOpType.DIV:
          this.div(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );

          break;
        case MLOpType.DOT:
          this.dot(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.SCALE:
          this.scale(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            this.params.get_element(params_start + 1)
          );
          break;
        case MLOpType.RELU:
          this.relu(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;
        case MLOpType.RELU_BACKWARD:
          this.relu_backward(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.TANH:
          this.tanh(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;
        case MLOpType.TANH_BACKWARD:
          this.tanh_backward(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.SIGMOID:
          this.sigmoid(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;
        case MLOpType.SIGMOID_BACKWARD:
          this.sigmoid_backward(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.MSE_LOSS:
          this.mse_loss(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)],
            this.params.get_element(params_start + 2),
            this.params.get_element(params_start + 3)
          );
          break;
        case MLOpType.MSE_LOSS_PRIME:
          this.mse_loss_prime(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.SOFTMAX:
          this.softmax(tensors[op.result], tensors[this.params.get_element(params_start)]);
          break;
        case MLOpType.SOFTMAX_BACKWARD:
          this.softmax_backward(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.CROSS_ENTROPY_LOSS:
          this.cross_entropy_loss(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)],
            this.params.get_element(params_start + 2),
            this.params.get_element(params_start + 3)
          );
          break;
        case MLOpType.CROSS_ENTROPY_LOSS_PRIME:
          this.cross_entropy_loss_prime(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.BINARY_CROSS_ENTROPY_LOSS:
          this.binary_cross_entropy_loss(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)],
            this.params.get_element(params_start + 2),
            this.params.get_element(params_start + 3)
          );
          break;
        case MLOpType.BINARY_CROSS_ENTROPY_LOSS_PRIME:
          this.binary_cross_entropy_loss_prime(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)]
          );
          break;
        case MLOpType.CLIP_L2_NORM:
          this.clip_l2_norm(tensors[op.result], this.params.get_element(params_start));
          break;
        case MLOpType.BATCH_REDUCE_MEAN:
          this.batch_reduce_mean(
            tensors[op.result],
            tensors[this.params.get_element(params_start)]
          );
          break;
        case MLOpType.ADAM_MOMENT_UPDATE:
          this.adam_moment_update(
            tensors[op.result],
            tensors[this.params.get_element(params_start)],
            tensors[this.params.get_element(params_start + 1)],
            tensors[this.params.get_element(params_start + 2)],
            this.params.get_element(params_start + 3),
            this.params.get_element(params_start + 4),
            this.params.get_element(params_start + 5),
            this.params.get_element(params_start + 6),
            this.params.get_element(params_start + 7)
          );
          break;
        case MLOpType.WRITE_TENSOR:
        case MLOpType.READ_TENSOR:
          tensors[op.result].data = tensors[this.params.get_element(params_start)].data;
          break;
        default:
          throw new Error(`Unknown operation type: ${op.type}`);
      }
    }
    return tensors;
  }
}

export class MLOpsGPU {
  static gpu_impl = null;

  static compile(ops, params) {
    this.gpu_impl.compile(ops, params);
  }

  static run() {
    return this.gpu_impl.run();
  }
}

export class MLOps {
  static backend = MLOpsCPU;
  static ops = new FrameAllocator(2048, MLOp);
  static params = new MLOpParams(2048 * 16);

  static set_backend(backend) {
    this.backend = backend;
  }

  static set_gpu_implementation(gpu_implementation) {
    if (this.backend.set_gpu_implementation) {
      this.backend.set_gpu_implementation(gpu_implementation);
    }
  }

  static reset() {
    this.ops.reset();
    this.params.reset();
    Tensor.cleanup();
  }

  static init_random(a, scale = 1) {
    const op = this.ops.allocate();
    op.type = MLOpType.INIT_RANDOM;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = a.id;
    this.params.add(scale, 1);
  }

  static init_he(a) {
    const op = this.ops.allocate();
    op.type = MLOpType.INIT_HE;
    op.param_start = this.params.length;
    op.param_count = 0;
    op.result = a.id;
  }

  static init_glorot(a, output_size) {
    const op = this.ops.allocate();
    op.type = MLOpType.INIT_GLOROT;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = a.id;
    this.params.add(output_size, 1);
  }

  static mat_mul(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([a.shape[0], b.shape[1]], a.batch_size);
    op.type = MLOpType.MAT_MUL;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    return result;
  }

  static transpose(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape.toReversed()], a.batch_size);
    op.type = MLOpType.TRANSPOSE;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static fill(a, value, offset = 0, size = null) {
    const op = this.ops.allocate();
    const result = a;
    op.type = MLOpType.FILL;
    op.param_start = this.params.length;
    op.param_count = 3;
    op.result = result.id;
    this.params.add(value, 1);
    this.params.add(offset, 1);
    this.params.add(size ?? a.length, 1);
    return result;
  }

  static extend(a, add_dims = [], fill_value = 0) {
    const op = this.ops.allocate();
    const new_shape = a.shape.map((dim, index) =>
      index < add_dims.length ? dim + add_dims[index] : dim
    );
    const result = Tensor.zeros(new_shape, a.batch_size);
    op.type = MLOpType.EXTEND;
    op.param_start = this.params.length;
    op.param_count = 2 + add_dims.length;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(add_dims, add_dims.length);
    this.params.add(fill_value, 1);
    return result;
  }

  static reshape(a, shape) {
    const op = this.ops.allocate();
    const result = Tensor.zeros(shape, a.batch_size);
    op.type = MLOpType.RESHAPE;
    op.param_start = this.params.length;
    op.param_count = 1 + shape.length;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(shape, shape.length);
    return result;
  }

  static copy(a, b, offset = 0, size = null) {
    const op = this.ops.allocate();
    const result = b;
    op.type = MLOpType.COPY;
    op.param_start = this.params.length;
    op.param_count = 3;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(offset, 1);
    this.params.add(size ?? a.length, 1);
    return result;
  }

  static clone(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.CLONE;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static add(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.ADD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    return result;
  }

  static sub(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SUB;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    return result;
  }

  static sub_assign(a, b) {
    const op = this.ops.allocate();
    const result = a;
    op.type = MLOpType.SUB_ASSIGN;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(b.id, 1);
    return result;
  }

  static div(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.DIV;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    return result;
  }

  static dot(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], a.batch_size);
    op.type = MLOpType.DOT;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    return result;
  }

  static scale(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SCALE;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b, 1);
    return result;
  }

  static relu(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.RELU;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static relu_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.RELU_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);
    return result;
  }

  static tanh(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.TANH;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static tanh_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.TANH_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);
    return result;
  }

  static sigmoid(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SIGMOID;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static sigmoid_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SIGMOID_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);
    return result;
  }

  static fused_mul_add(a, b, c) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.FUSED_MUL_ADD;
    op.param_start = this.params.length;
    op.param_count = 3;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    this.params.add(c.id, 1);
    return result;
  }

  static mse_loss(target, output, enabled_logging = false, name = null) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], target.batch_size);
    op.type = MLOpType.MSE_LOSS;
    op.param_start = this.params.length;
    op.param_count = 4;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    this.params.add(enabled_logging, 1);
    this.params.add(name, 1);
    return result;
  }

  static mse_loss_prime(target, output) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...target.shape], target.batch_size);
    op.type = MLOpType.MSE_LOSS_PRIME;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    return result;
  }

  static softmax(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SOFTMAX;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static softmax_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SOFTMAX_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);
    return result;
  }

  static cross_entropy_loss(target, output, enabled_logging = false, name = null) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], target.batch_size);
    op.type = MLOpType.CROSS_ENTROPY_LOSS;
    op.param_start = this.params.length;
    op.param_count = 4;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    this.params.add(enabled_logging, 1);
    this.params.add(name, 1);
    return result;
  }

  static cross_entropy_loss_prime(target, output) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...target.shape], target.batch_size);
    op.type = MLOpType.CROSS_ENTROPY_LOSS_PRIME;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    return result;
  }

  static binary_cross_entropy_loss(target, output, enabled_logging = false, name = null) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], target.batch_size);
    op.type = MLOpType.BINARY_CROSS_ENTROPY_LOSS;
    op.param_start = this.params.length;
    op.param_count = 4;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    this.params.add(enabled_logging, 1);
    this.params.add(name, 1);
    return result;
  }

  static binary_cross_entropy_loss_prime(target, output) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...target.shape], target.batch_size);
    op.type = MLOpType.BINARY_CROSS_ENTROPY_LOSS_PRIME;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    return result;
  }

  static clip_l2_norm(a, max_norm) {
    const op = this.ops.allocate();
    const result = a;
    op.type = MLOpType.CLIP_L2_NORM;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(max_norm, 1);
    return result;
  }

  static batch_reduce_mean(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape]);
    op.type = MLOpType.BATCH_REDUCE_MEAN;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);
    return result;
  }

  static adam_moment_update(
    variable,
    m_tensor,
    v_tensor,
    grad,
    beta1,
    beta2,
    t,
    epsilon,
    learning_rate
  ) {
    const op = this.ops.allocate();
    const result = variable;
    op.type = MLOpType.ADAM_MOMENT_UPDATE;
    op.param_start = this.params.length;
    op.param_count = 8;
    op.result = result.id;
    this.params.add(m_tensor.id, 1);
    this.params.add(v_tensor.id, 1);
    this.params.add(grad.id, 1);
    this.params.add(beta1, 1);
    this.params.add(beta2, 1);
    this.params.add(t, 1);
    this.params.add(epsilon, 1);
    this.params.add(learning_rate, 1);
    return result;
  }

  static write_tensor(tensor, index, value) {
    const op = this.ops.allocate();
    const result = tensor;
    op.type = MLOpType.WRITE_TENSOR;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(index, 1);
    this.params.add(value, 1);
    return result;
  }

  static read_tensor(tensor, index) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...tensor.shape], tensor.batch_size);
    op.type = MLOpType.READ_TENSOR;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(tensor.id, 1);
    this.params.add(index, 1);
    return result;
  }

  static compile() {
    this.backend.compile(this.ops, this.params);
  }

  static run() {
    return this.backend.run();
  }
}
