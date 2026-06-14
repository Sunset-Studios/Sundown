import { InputType } from "../../ml_types.js";
import { Tensor } from "../../math/tensor.js";
import {
  BaseDataProvider,
  DataProviderKind,
  require_number,
  require_positive_integer,
} from "../data_provider.js";

const math_names = Object.getOwnPropertyNames(Math).filter((name) => name !== "random");

function compile_expression(expression) {
  if (typeof expression === "function") {
    return expression;
  }

  if (typeof expression !== "string" || expression.trim().length === 0) {
    throw new Error("FormulaProvider requires a non-empty expression.");
  }

  const fn = new Function(
    "sample",
    "x",
    "index",
    "batch_index",
    "input",
    "target",
    "random",
    "Math",
    ...math_names,
    `"use strict"; return (${expression});`
  );

  return (sample) =>
    fn(
      sample,
      sample.x,
      sample.index,
      sample.batch_index,
      sample.input,
      sample.target,
      sample.random,
      Math,
      ...math_names.map((name) => Math[name])
    );
}

function write_value(data, offset, size, value) {
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    if (value.length !== size) {
      throw new Error(`Formula result length ${value.length} does not match shape size ${size}.`);
    }
    for (let i = 0; i < size; i++) {
      data[offset + i] = require_number(Number(value[i]), "formula result");
    }
    return;
  }

  if (size !== 1) {
    throw new Error(`Scalar formula results require shape size 1, received shape size ${size}.`);
  }

  data[offset] = require_number(Number(value), "formula result");
}

export class FormulaProvider extends BaseDataProvider {
  constructor(options = {}) {
    super({
      kind: DataProviderKind.FORMULA,
      label: options.label ?? "Formula",
      input_type: InputType.NUMERIC,
      shape: options.shape,
      batch_size: options.batch_size,
      finite: options.finite,
      loop: options.loop,
    });

    this.expression = options.expression;
    this.formula = compile_expression(this.expression);
    this.count = options.finite ? require_positive_integer(options.count, "FormulaProvider.count") : Infinity;
    this.index = 0;
    this.min = require_number(options.min ?? -1.0, "FormulaProvider.min");
    this.max = require_number(options.max ?? 1.0, "FormulaProvider.max");
  }

  snapshot() {
    return { index: this.index, state: this.state };
  }

  restore(snapshot) {
    this.index = snapshot.index;
    this.state = snapshot.state;
  }

  reset() {
    super.reset();
    this.index = 0;
  }

  next(count = this.batch_size, context = {}) {
    if (this.is_paused()) {
      return null;
    }

    if (this.index >= this.count && !this.loop) {
      this.state = "exhausted";
      return null;
    }

    const available_count = this.loop ? count : Math.min(count, this.count - this.index);
    if (available_count <= 0) {
      this.state = "exhausted";
      return null;
    }

    const size = Tensor.sample_size(this.shape);
    const data = new Float32Array(size * available_count);
    const input_data = context.input_batch ? context.input_batch.data : context.input_tensor?.data;
    const input_shape = context.input_batch ? context.input_batch.shape : context.input_tensor?.shape;
    const input_size = input_shape ? Tensor.sample_size(input_shape) : 0;

    const samples = [];
    for (let batch_index = 0; batch_index < available_count; batch_index++) {
      const sample_index = this.index + batch_index;
      const random_value = this.min + Math.random() * (this.max - this.min);
      const input = input_data && input_size > 0
        ? input_data.subarray(batch_index * input_size, (batch_index + 1) * input_size)
        : null;
      const sample = {
        index: sample_index,
        batch_index,
        random: random_value,
        x: input ? input[0] : random_value,
        input,
        input_shape,
      };
      const value = this.formula(sample);
      write_value(data, batch_index * size, size, value);
      samples.push({ index: sample_index, input: input ? Array.from(input) : null, value });
    }

    this.index += available_count;
    if (this.index >= this.count && !this.loop) {
      this.state = "exhausted";
    }

    return {
      data,
      shape: this.shape,
      batch_size: available_count,
      input_type: this.input_type,
      meta: { samples },
      done: this.state === "exhausted",
    };
  }
}
