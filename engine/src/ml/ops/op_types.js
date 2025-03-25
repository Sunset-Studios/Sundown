// An enum of all supported low-level operations.
export const MLOpType = Object.freeze({
  NONE: 0,
  CREATE_ZERO_TENSOR: 1,
  INIT_RANDOM: 2,
  INIT_HE: 3,
  INIT_GLOROT: 4,
  MAT_MUL: 5,
  TRANSPOSE: 6,
  FILL: 7,
  EXTEND: 8,
  RESHAPE: 9,
  COPY: 10,
  CLONE: 11,
  ADD: 12,
  SUB: 13,
  SUB_ASSIGN: 14,
  DIV: 15,
  SCALE: 16,
  FUSED_MUL_ADD: 17,
  RELU: 18,
  RELU_BACKWARD: 19,
  TANH: 20,
  TANH_BACKWARD: 21,
  SIGMOID: 22,
  SIGMOID_BACKWARD: 23,
  MSE_LOSS: 24,
  MSE_LOSS_PRIME: 25,
  SOFTMAX: 26,
  SOFTMAX_BACKWARD: 27,
  CROSS_ENTROPY_LOSS: 28,
  CROSS_ENTROPY_LOSS_PRIME: 29,
  BINARY_CROSS_ENTROPY_LOSS: 30,
  BINARY_CROSS_ENTROPY_LOSS_PRIME: 31,
  CLIP_L2_NORM: 32,
  BATCH_REDUCE_MEAN: 33,
  ADAM_MOMENT_UPDATE: 34,
  WRITE_TENSOR: 35,
  READ_TENSOR: 36,
});

// An enum of all supported high-level operations.
export const MLHopType = Object.freeze({
  NONE: 0,
  CREATE_MODEL: 1,
  ADD_LAYER: 2,
  ADD_ACTIVATION: 3,
  ADD_LOSS: 4,
  RESET_MODEL: 5,
});

// A class that represents a low-level operation and all associated parameters.
export class MLOp {
  type = MLOpType.NONE;
  param_start = 0;
  param_count = 0;
  result = -1;
}

// A class that represents the parameter list of a low-level operation.
export class MLOpParams {
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

  resize_if_necessary(new_size) {
    if (new_size > this.#capacity) {
      this.#capacity = new_size * 2;
      const new_data = new Float64Array(this.#capacity);
      new_data.set(this.data);
      this.data = new_data;
    }
  }

  add(data, size = null) {
    let true_size;

    if (Array.isArray(data) || ArrayBuffer.isView(data)) {
      true_size = size ?? data.length;
      this.resize_if_necessary(this.#size + true_size);
      this.data.set(data, this.#size);
      this.#size += true_size;
    } else {
      true_size = 1;
      this.resize_if_necessary(this.#size + true_size);
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

  append(other) {
    this.resize_if_necessary(this.#size + other.#size);
    this.data.set(other.data, this.#size);
    this.#size += other.#size;
  }

  get length() {
    return this.#size;
  }
}

// A class that represents a high-level operation, usually encapsulating an API object.
export class MLHop {
  type = MLHopType.NONE;
  result = null;
}