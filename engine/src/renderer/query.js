import { Buffer } from "./buffer.js";

const QueryType = {
  Timestamp: "timestamp",
  Occlusion: "occlusion",
};

const default_capacity = 1024;

export class GPUTimeQuery {
  static query_set = null;
  static query_buffer = null;
  static current_query_index = 0;
  static latest_results = new BigInt64Array(default_capacity);

  static init(device) {
    this.device = device;

    // Create the query set for timestamps (if supported)
    if (!GPUTimeQuery.query_set) {
      GPUTimeQuery.query_set = device.createQuerySet({
        type: QueryType.Timestamp,
        count: default_capacity,
      });
    }

    // Buffer to hold resolved timestamps (8 bytes per timestamp)
    if (!GPUTimeQuery.query_buffer) {
      GPUTimeQuery.query_buffer = Buffer.create({
        name: "query_buffer",
        size: default_capacity * 8,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_DST,
        cpu_readback: true,
      });
    }

    this.reset();
  }

  // Reset state for a new frame
  static reset() {
    GPUTimeQuery.current_query_index = 0;
  }

  

  // Resolve query results into the buffer at the end of encoding
  static resolve(encoder) {
    if (GPUTimeQuery.current_query_index === 0) return;
    encoder.resolveQuerySet(
      GPUTimeQuery.query_set,
      0,
      GPUTimeQuery.query_set.count,
      GPUTimeQuery.query_buffer.buffer,
      0
    );
  }

  // Read back and process the resolved timestamps. Should be called after GPU work is done.
  static async read() {
    await GPUTimeQuery.query_buffer.read(
      GPUTimeQuery.latest_results,
      default_capacity * 8,
      0,
      0,
      BigInt64Array
    );
    return GPUTimeQuery.latest_results;
  }

  // Get the last processed results
  static get_results() {
    return GPUTimeQuery.latest_results;
  }

  static allocate() {
    const query_index = GPUTimeQuery.current_query_index;
    GPUTimeQuery.current_query_index += 1;
    return query_index;
  }
}
