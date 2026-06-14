import { InputType } from "../../ml_types.js";
import { Tensor } from "../../math/tensor.js";
import {
  BaseDataProvider,
  DataProviderKind,
  require_number,
  require_positive_integer,
} from "../data_provider.js";

function random_normal() {
  const u = Math.max(Number.EPSILON, Math.random());
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export class NoiseProvider extends BaseDataProvider {
  constructor(options = {}) {
    const finite = options.count !== undefined;

    super({
      kind: DataProviderKind.NOISE,
      label: options.label ?? "Noise",
      input_type: InputType.NUMERIC,
      shape: options.shape,
      batch_size: options.batch_size,
      finite: options.count !== undefined,
      loop: options.loop,
    });

    this.min = require_number(options.min ?? -1.0, "NoiseProvider.min");
    this.max = require_number(options.max ?? 1.0, "NoiseProvider.max");
    this.mean = require_number(options.mean ?? 0.0, "NoiseProvider.mean");
    this.stddev = require_number(options.stddev ?? 1.0, "NoiseProvider.stddev");
    this.distribution = options.distribution ?? "uniform";
    if (this.distribution !== "uniform" && this.distribution !== "normal") {
      throw new Error("NoiseProvider.distribution must be 'uniform' or 'normal'.");
    }
    this.count = finite ? require_positive_integer(options.count, "NoiseProvider.count") : Infinity;
    this.index = 0;
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

  next(count = this.batch_size) {
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

    for (let i = 0; i < data.length; i++) {
      if (this.distribution === "normal") {
        data[i] = this.mean + random_normal() * this.stddev;
      } else {
        data[i] = this.min + Math.random() * (this.max - this.min);
      }
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
      meta: { start_index: this.index - available_count },
      done: this.state === "exhausted",
    };
  }
}
