import { InputType } from "../../ml_types.js";
import { Tensor } from "../../math/tensor.js";
import {
  BaseDataProvider,
  DataProviderKind,
  require_positive_integer,
  require_shape,
} from "../data_provider.js";

function unique_chars(text) {
  return Array.from(new Set(Array.from(text)));
}

export class TextStreamProvider extends BaseDataProvider {
  constructor(options = {}) {
    if (typeof options.text !== "string" || options.text.length === 0) {
      throw new Error("TextStreamProvider.text must be a non-empty string.");
    }

    const text = options.text;
    const vocab = options.vocab ?? unique_chars(text);
    if (!Array.isArray(vocab) || vocab.length === 0) {
      throw new Error("TextStreamProvider.vocab must be a non-empty array.");
    }

    const context_length = require_positive_integer(
      options.context_length,
      "TextStreamProvider.context_length"
    );
    const mode = options.mode ?? "context";
    if (mode !== "context" && mode !== "target") {
      throw new Error("TextStreamProvider.mode must be 'context' or 'target'.");
    }

    const encoding = options.encoding ?? "one_hot";
    if (encoding !== "one_hot" && encoding !== "index") {
      throw new Error("TextStreamProvider.encoding must be 'one_hot' or 'index'.");
    }

    const shape = options.shape ?? (
      encoding === "one_hot"
        ? mode === "target" ? [vocab.length] : [context_length, vocab.length]
        : mode === "target" ? [1] : [context_length]
    );
    const stride = require_positive_integer(options.stride, "TextStreamProvider.stride");
    const start = options.start ?? 0;
    if (!Number.isInteger(start) || start < 0) {
      throw new Error("TextStreamProvider.start must be a non-negative integer.");
    }

    super({
      kind: DataProviderKind.TEXT,
      label: options.label ?? "Text Stream",
      input_type: InputType.TEXT,
      shape: require_shape(shape, "TextStreamProvider.shape"),
      batch_size: options.batch_size,
      finite: true,
      loop: !!options.loop,
    });

    this.text = text;
    this.vocab = vocab;
    this.token_to_index = new Map(vocab.map((token, index) => [token, index]));
    this.context_length = context_length;
    this.mode = mode;
    this.encoding = encoding;
    this.stride = stride;
    this.cursor = start;
  }

  snapshot() {
    return { cursor: this.cursor, state: this.state };
  }

  restore(snapshot) {
    this.cursor = snapshot.cursor;
    this.state = snapshot.state;
  }

  reset() {
    super.reset();
    this.cursor = 0;
  }

  get max_cursor() {
    return Math.max(0, this.text.length - this.context_length);
  }

  write_token(data, offset, token) {
    if (this.encoding === "one_hot") {
      if (!this.token_to_index.has(token)) {
        throw new Error(`TextStreamProvider token is not in vocab: ${token}`);
      }
      const index = this.token_to_index.get(token);
      data[offset + index] = 1.0;
      return;
    }

    if (!this.token_to_index.has(token)) {
      throw new Error(`TextStreamProvider token is not in vocab: ${token}`);
    }
    data[offset] = this.token_to_index.get(token);
  }

  next(count = this.batch_size) {
    if (this.is_paused()) {
      return null;
    }

    if (this.text.length === 0 || (this.cursor >= this.max_cursor && !this.loop)) {
      this.state = "exhausted";
      return null;
    }

    const sample_size = Tensor.sample_size(this.shape);
    const available_count = this.loop ? count : Math.min(count, this.max_cursor - this.cursor);
    const data = new Float32Array(sample_size * available_count);
    const samples = [];

    for (let batch_index = 0; batch_index < available_count; batch_index++) {
      if (this.cursor >= this.max_cursor && this.loop) {
        this.cursor = 0;
      }

      const start = this.cursor;
      const context = this.text.slice(start, start + this.context_length);
      const target = this.text[start + this.context_length];

      if (this.mode === "target") {
        this.write_token(data, batch_index * sample_size, target);
      } else {
        for (let i = 0; i < this.context_length; i++) {
          const token = context[i];
          const offset = this.encoding === "one_hot"
            ? batch_index * sample_size + i * this.vocab.length
            : batch_index * sample_size + i;
          this.write_token(data, offset, token);
        }
      }

      samples.push({ start, context, target });
      this.cursor += this.stride;
    }

    if (this.cursor >= this.max_cursor && !this.loop) {
      this.state = "exhausted";
    }

    return {
      data,
      shape: this.shape,
      batch_size: available_count,
      input_type: this.input_type,
      meta: { samples, vocab: this.vocab },
      done: this.state === "exhausted",
    };
  }
}
