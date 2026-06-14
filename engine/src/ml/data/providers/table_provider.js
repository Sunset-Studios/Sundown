import { InputType } from "../../ml_types.js";
import { Tensor } from "../../math/tensor.js";
import {
  BaseDataProvider,
  DataProviderKind,
} from "../data_provider.js";

function parse_csv_line(line) {
  const result = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && quoted && next === '"') {
      current += '"';
      i++;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
}

function normalize_rows(rows, has_headers, headers = null) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error("TableProvider.rows must be a non-empty array.");
  }

  if (typeof has_headers !== "boolean") {
    throw new Error("TableProvider.has_headers must be a boolean.");
  }

  if (!Array.isArray(rows[0])) {
    const object_headers = headers ?? Object.keys(rows[0]);
    if (object_headers.length === 0) {
      throw new Error("TableProvider object rows require at least one field.");
    }
    return {
      headers: object_headers,
      rows: rows.map((row) => object_headers.map((header) => row[header])),
    };
  }

  if (headers) {
    return { rows, headers };
  }

  if (!has_headers) {
    return { rows, headers: rows[0].map((_, index) => index) };
  }

  return {
    headers: rows[0],
    rows: rows.slice(1),
  };
}

function resolve_column_indices(columns, headers) {
  if (!columns || columns.length === 0) {
    throw new Error("TableProvider.columns must list at least one column.");
  }

  return columns.map((column) => {
    const index = typeof column === "number" ? column : headers.indexOf(column);
    if (index < 0 || index >= headers.length) {
      throw new Error(`TableProvider column not found: ${column}`);
    }
    return index;
  });
}

export class TableProvider extends BaseDataProvider {
  constructor(options = {}) {
    const normalized = normalize_rows(options.rows, options.has_headers ?? true, options.headers);
    const column_indices = resolve_column_indices(options.columns, normalized.headers);
    const start = options.start ?? 0;
    if (!Number.isInteger(start) || start < 0 || start > normalized.rows.length) {
      throw new Error("TableProvider.start must be an integer within the row range.");
    }

    super({
      kind: DataProviderKind.TABLE,
      label: options.label ?? "Table",
      input_type: InputType.NUMERIC,
      shape: options.shape ?? [column_indices.length],
      batch_size: options.batch_size,
      finite: true,
      loop: options.loop,
    });

    this.rows = normalized.rows;
    this.headers = normalized.headers;
    this.column_indices = column_indices;
    this.cursor = start;
    this.shuffle = !!options.shuffle;
    this.order = this.rows.map((_, index) => index);

    if (this.shuffle) {
      this.shuffle_order();
    }
  }

  static from_csv(text, options = {}) {
    if (typeof text !== "string" || text.length === 0) {
      throw new Error("TableProvider.from_csv requires non-empty CSV text.");
    }

    const rows = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map(parse_csv_line);

    return new TableProvider({ ...options, rows });
  }

  snapshot() {
    return {
      cursor: this.cursor,
      order: this.order.slice(),
      state: this.state,
    };
  }

  restore(snapshot) {
    this.cursor = snapshot.cursor;
    this.order = snapshot.order.slice();
    this.state = snapshot.state;
  }

  reset() {
    super.reset();
    this.cursor = 0;
    this.order = this.rows.map((_, index) => index);
    if (this.shuffle) {
      this.shuffle_order();
    }
  }

  shuffle_order() {
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
  }

  next(count = this.batch_size) {
    if (this.is_paused()) {
      return null;
    }

    if (this.rows.length === 0 || (this.cursor >= this.rows.length && !this.loop)) {
      this.state = "exhausted";
      return null;
    }

    const expected_column_count = Tensor.sample_size(this.shape);
    if (expected_column_count !== this.column_indices.length) {
      throw new Error(
        `TableProvider shape size ${expected_column_count} does not match ${this.column_indices.length} columns.`
      );
    }

    const row_sample_size = expected_column_count;
    const available_count = this.loop ? count : Math.min(count, this.rows.length - this.cursor);
    const data = new Float32Array(row_sample_size * available_count);
    const row_indices = [];

    for (let batch_index = 0; batch_index < available_count; batch_index++) {
      if (this.cursor >= this.rows.length && this.loop) {
        this.cursor = 0;
        if (this.shuffle) {
          this.shuffle_order();
        }
      }

      const row_index = this.order[this.cursor++];
      const row = this.rows[row_index];
      row_indices.push(row_index);

      for (let i = 0; i < row_sample_size; i++) {
        const column_index = this.column_indices[i];
        const value = Number(row[column_index]);
        if (!Number.isFinite(value)) {
          throw new Error(`TableProvider row ${row_index} column ${column_index} is not numeric.`);
        }
        data[batch_index * row_sample_size + i] = value;
      }
    }

    if (this.cursor >= this.rows.length && !this.loop) {
      this.state = "exhausted";
    }

    return {
      data,
      shape: this.shape,
      batch_size: available_count,
      input_type: this.input_type,
      meta: { row_indices },
      done: this.state === "exhausted",
    };
  }
}
