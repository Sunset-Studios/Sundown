import { InputType } from "../ml_types.js";
import { Tensor } from "../math/tensor.js";

export const DataChannel = Object.freeze({
  INPUT: "input",
  TARGET: "target",
});

export const DataProviderKind = Object.freeze({
  FORMULA: "formula",
  TABLE: "table",
  TEXT: "text",
  IMAGE_FOLDER: "image_folder",
  NOISE: "noise",
});

export const DataStreamState = Object.freeze({
  READY: "ready",
  EXHAUSTED: "exhausted",
  PAUSED: "paused",
  ERROR: "error",
});

function validate_provider_options(options) {
  if (!options.kind) {
    throw new Error("Data provider requires a kind.");
  }
  require_input_type(options.input_type);
  require_shape(options.shape);
  require_positive_integer(options.batch_size, "batch_size");
  require_boolean(options.finite, "finite");
  require_boolean(options.loop, "loop");
}

function validate_tensor_batch(batch) {
  if (!batch.tensor.data || !batch.tensor.shape || !batch.tensor.batch_size) {
    throw new Error("Provider tensor batches require tensor data, shape, and batch_size.");
  }
  require_input_type(batch.input_type);
}

function validate_data_batch(batch) {
  require_shape(batch.shape);
  require_positive_integer(batch.batch_size, "batch_size");
  require_input_type(batch.input_type);

  if (!ArrayBuffer.isView(batch.data)) {
    throw new Error("Provider batch data must be a typed array.");
  }

  const expected_length = Tensor.sample_size(batch.shape) * batch.batch_size;
  if (batch.data.length !== expected_length) {
    throw new Error(
      `Provider batch data length ${batch.data.length} does not match expected length ${expected_length}.`
    );
  }
}

export function require_shape(shape, label = "shape") {
  const normalized_shape = Tensor.normalize_shape(shape);
  if (
    !normalized_shape ||
    normalized_shape.length === 0 ||
    normalized_shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)
  ) {
    throw new Error(`${label} must be a non-empty array of positive integers.`);
  }

  return normalized_shape;
}

export function require_positive_integer(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

export function require_number(value, label) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return value;
}

export function require_input_type(value, label = "input_type") {
  if (!Object.values(InputType).includes(value)) {
    throw new Error(`${label} must be a valid InputType.`);
  }
  return value;
}

export function require_boolean(value, label) {
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be a boolean.`);
  }
  return value;
}

export function get_provider_shape(provider) {
  if (!provider) {
    return null;
  }
  const description = provider.describe();
  return Tensor.normalize_shape(description.shape);
}

export function normalize_provider_batch(batch) {
  if (!batch) {
    return null;
  }

  if (batch.tensor) {
    validate_tensor_batch(batch);
    return {
      tensor: batch.tensor,
      data: batch.tensor.data,
      shape: batch.tensor.shape,
      batch_size: batch.tensor.batch_size,
      input_type: batch.input_type,
      meta: batch.meta ?? null,
      done: batch.done === true,
    };
  } else {
    validate_data_batch(batch);
    const shape = Tensor.normalize_shape(batch.shape);
    const data = batch.data;
    const ArrayType = data.constructor;
    return {
      data,
      shape,
      batch_size: batch.batch_size,
      ArrayType,
      input_type: batch.input_type,
      meta: batch.meta ?? null,
      done: batch.done === true,
    };
  }
}

export class DataProviderRegistry {
  static providers = new Map();

  static register(kind, provider_class) {
    if (!kind || !provider_class) {
      throw new Error("DataProviderRegistry.register requires a kind and provider class.");
    }
    DataProviderRegistry.providers.set(kind, provider_class);
  }

  static create(kind, options = {}) {
    const provider_class = DataProviderRegistry.providers.get(kind);
    if (!provider_class) {
      throw new Error(`Unknown data provider kind: ${kind}`);
    }
    return new provider_class(options);
  }

  static has(kind) {
    return DataProviderRegistry.providers.has(kind);
  }
}

export class BaseDataProvider {
  constructor(options = {}) {
    validate_provider_options(options);

    this.kind = options.kind;
    this.label = options.label ?? options.kind;
    this.input_type = options.input_type;
    this.shape = Tensor.normalize_shape(options.shape);
    this.batch_size = options.batch_size;
    this.finite = options.finite;
    this.loop = options.loop;
    this.paused = false;
    this.state = DataStreamState.READY;
  }

  describe() {
    return {
      kind: this.kind,
      label: this.label,
      input_type: this.input_type,
      shape: this.shape,
      batch_size: this.batch_size,
      finite: this.finite,
      loop: this.loop,
      paused: this.paused,
      state: this.paused ? DataStreamState.PAUSED : this.state,
    };
  }

  reset() {
    this.paused = false;
    this.state = DataStreamState.READY;
  }

  pause() {
    this.paused = true;
  }

  resume() {
    this.paused = false;
  }

  is_paused() {
    return this.paused;
  }

  next(count = this.batch_size, context = {}) {
    throw new Error(`${this.constructor.name}.next() must be implemented.`);
  }

  preview(count = this.batch_size, context = {}) {
    if (typeof this.snapshot !== "function" || typeof this.restore !== "function") {
      throw new Error(`${this.constructor.name}.preview() requires snapshot() and restore().`);
    }

    const snapshot = this.snapshot();
    const result = this.next(count, { ...context, preview: true });
    this.restore(snapshot);
    return result;
  }
}
