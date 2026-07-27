export const StreamRequestStatus = Object.freeze({
  STARTING: "starting",
  STREAMING: "streaming",
  COMPLETED: "completed",
  FAILED: "failed",
  CANCELLED: "cancelled",
});

export const StreamUpdateStatus = Object.freeze({
  PENDING: "pending",
  CONTINUE: "continue",
  COMPLETE: "complete",
  CANCEL: "cancel",
});

/**
 * Base contract for a streaming implementation.
 *
 * Providers own domain-specific acquisition, serialization, and incremental
 * commit logic. StreamingSystem owns request routing and lifecycle state.
 */
export class StreamProvider {
  static provider_type = null;

  constructor(options = {}) {
    this.provider_type = options.provider_type ?? this.constructor.provider_type;
    if (typeof this.provider_type !== "string" || this.provider_type.length === 0) {
      throw new Error(`${this.constructor.name} requires a non-empty provider_type.`);
    }
  }

  /**
   * Creates provider state for a new request. May return a state object or a
   * promise for one.
   */
  begin_stream(_request, _context = {}) {
    return null;
  }

  /**
   * Creates state shared by this provider's requests during one system update.
   * Providers can use this for bandwidth, upload, or work-item budgets.
   */
  begin_frame(_context = {}) {
    return null;
  }

  /**
   * Advances one request. This hook must remain synchronous so work stays
   * bounded by the current frame; asynchronous acquisition belongs in
   * begin_stream() and can be polled through provider state.
   */
  update_stream(_request, _context = {}) {
    throw new Error(`${this.constructor.name}.update_stream() must be implemented.`);
  }

  complete_stream(_request, _context = {}) {}

  cancel_stream(_request, _context = {}) {}

  cleanup(_context = {}) {}

  serialize(_value, _context = {}) {
    throw new Error(`${this.constructor.name}.serialize() must be implemented.`);
  }

  deserialize(_payload, _context = {}) {
    throw new Error(`${this.constructor.name}.deserialize() must be implemented.`);
  }
}
