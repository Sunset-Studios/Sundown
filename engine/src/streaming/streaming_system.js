import { SimulationLayer } from "../core/simulation_layer.js";
import { StreamProvider, StreamRequestStatus, StreamUpdateStatus } from "./stream_provider.js";

function is_terminal_status(status) {
  return (
    status === StreamRequestStatus.COMPLETED ||
    status === StreamRequestStatus.FAILED ||
    status === StreamRequestStatus.CANCELLED
  );
}

function normalize_update_result(result) {
  if (result === undefined || result === null) {
    return { status: StreamUpdateStatus.CONTINUE, result: undefined };
  }

  if (typeof result === "string") {
    return { status: result, result: undefined };
  }

  return {
    status: result.status ?? StreamUpdateStatus.CONTINUE,
    result: result.result,
  };
}

export class StreamRequest {
  constructor(system, id, provider_type, target, options) {
    this.system = system;
    this.id = id;
    this.provider_type = provider_type;
    this.target = target;
    this.options = options;
    this.status = StreamRequestStatus.STARTING;
    this.state = null;
    this.result = null;
    this.error = null;

    this.finished = new Promise((resolve) => {
      this._resolve_finished = resolve;
    });
  }

  cancel(reason = "Stream request cancelled") {
    return this.system.cancel(this, reason);
  }

  get is_terminal() {
    return is_terminal_status(this.status);
  }
}

/**
 * Provider-driven streaming coordinator shared by renderer, gameplay, and
 * other engine systems.
 */
export class StreamingSystem extends SimulationLayer {
  static instance = null;

  constructor() {
    super();
    this.name = "StreamingSystem";
    this.providers = new Map();
    this.requests = new Map();
    this.next_request_id = 1;
  }

  register_provider(provider_or_class, options = {}) {
    const provider =
      typeof provider_or_class === "function" ? new provider_or_class(options) : provider_or_class;

    if (!(provider instanceof StreamProvider)) {
      throw new Error("StreamingSystem providers must extend StreamProvider.");
    }

    if (this.providers.has(provider.provider_type) && !options.replace) {
      throw new Error(`Streaming provider '${provider.provider_type}' is already registered.`);
    }

    if (this.providers.has(provider.provider_type)) {
      this.unregister_provider(provider.provider_type);
    }

    this.providers.set(provider.provider_type, provider);
    return provider;
  }

  unregister_provider(provider_type) {
    const provider = this.providers.get(provider_type);
    if (!provider) {
      return false;
    }

    for (const request of Array.from(this.requests.values())) {
      if (request.provider_type === provider_type) {
        this.cancel(request, `Streaming provider '${provider_type}' was unregistered.`);
      }
    }

    provider.cleanup({ system: this });
    this.providers.delete(provider_type);
    return true;
  }

  has_provider(provider_type) {
    return this.providers.has(provider_type);
  }

  get_provider(provider_type) {
    return this.providers.get(provider_type) ?? null;
  }

  stream(provider_type, target, options = {}) {
    const provider = this.providers.get(provider_type);
    if (!provider) {
      throw new Error(`Streaming provider '${provider_type}' is not registered.`);
    }

    const request = new StreamRequest(this, this.next_request_id++, provider_type, target, options);
    this.requests.set(request.id, request);

    try {
      const provider_state = provider.begin_stream(request, { system: this });
      if (provider_state && typeof provider_state.then === "function") {
        provider_state.then(
          (state) => this._finish_start(request, provider, state),
          (stream_error) => this._fail(request, stream_error)
        );
      } else {
        request.state = provider_state;
        request.status = StreamRequestStatus.STREAMING;
      }
    } catch (stream_error) {
      this._fail(request, stream_error);
    }

    return request;
  }

  update(delta_time) {
    super.update(delta_time);
    this.process_streams(delta_time);
  }

  process_streams(delta_time = 0) {
    for (const provider of this.providers.values()) {
      const provider_requests = Array.from(this.requests.values()).filter(
        (request) =>
          request.provider_type === provider.provider_type &&
          request.status === StreamRequestStatus.STREAMING
      );
      if (provider_requests.length === 0) {
        continue;
      }

      let frame = null;
      try {
        frame = provider.begin_frame({
          delta_time,
          request_count: provider_requests.length,
          system: this,
        });
      } catch (stream_error) {
        for (const request of provider_requests) {
          this._fail(request, stream_error);
        }
        continue;
      }

      for (const request of provider_requests) {
        if (request.status !== StreamRequestStatus.STREAMING) {
          continue;
        }

        try {
          const update_result = provider.update_stream(request, {
            delta_time,
            frame,
            system: this,
          });
          if (update_result && typeof update_result.then === "function") {
            throw new Error(
              `${provider.constructor.name}.update_stream() returned a promise; ` +
                "incremental stream updates must be synchronous."
            );
          }

          const normalized_result = normalize_update_result(update_result);
          switch (normalized_result.status) {
            case StreamUpdateStatus.PENDING:
            case StreamUpdateStatus.CONTINUE:
              break;
            case StreamUpdateStatus.COMPLETE:
              this._complete(request, normalized_result.result);
              break;
            case StreamUpdateStatus.CANCEL:
              this.cancel(request);
              break;
            default:
              throw new Error(
                `${provider.constructor.name}.update_stream() returned unknown status ` +
                  `'${normalized_result.status}'.`
              );
          }
        } catch (stream_error) {
          this._fail(request, stream_error);
        }
      }
    }
  }

  cancel(request_or_id, reason = "Stream request cancelled") {
    const request =
      typeof request_or_id === "number" ? this.requests.get(request_or_id) : request_or_id;
    if (!request || is_terminal_status(request.status)) {
      return false;
    }

    const provider = this.providers.get(request.provider_type);
    try {
      provider?.cancel_stream(request, {
        reason,
        system: this,
      });
    } catch (cancel_error) {
      request.error = cancel_error;
    }

    request.status = StreamRequestStatus.CANCELLED;
    request.error ??= new Error(reason);
    this._remove_request(request);
    return true;
  }

  cancel_target(provider_type, target, reason = "Stream target cancelled") {
    let cancelled = false;
    for (const request of Array.from(this.requests.values())) {
      if (request.provider_type === provider_type && request.target === target) {
        cancelled = this.cancel(request, reason) || cancelled;
      }
    }
    return cancelled;
  }

  find_requests(provider_type, target = undefined) {
    return Array.from(this.requests.values()).filter(
      (request) =>
        request.provider_type === provider_type &&
        (target === undefined || request.target === target)
    );
  }

  serialize(provider_type, value, context = {}) {
    const provider = this.providers.get(provider_type);
    if (!provider) {
      throw new Error(`Streaming provider '${provider_type}' is not registered.`);
    }
    return provider.serialize(value, { ...context, system: this });
  }

  deserialize(provider_type, payload, context = {}) {
    const provider = this.providers.get(provider_type);
    if (!provider) {
      throw new Error(`Streaming provider '${provider_type}' is not registered.`);
    }
    return provider.deserialize(payload, { ...context, system: this });
  }

  cleanup() {
    for (const request of Array.from(this.requests.values())) {
      this.cancel(request, "Streaming system shutdown");
    }
    for (const provider of this.providers.values()) {
      provider.cleanup({ system: this });
    }
    this.providers.clear();
    super.cleanup();
  }

  _finish_start(request, provider, state) {
    if (is_terminal_status(request.status)) {
      request.state = state;
      try {
        provider.cancel_stream(request, {
          reason: "Stream request ended before asynchronous startup completed.",
          system: this,
        });
      } catch {
        // The request already reached its terminal state.
      }
      return;
    }

    request.state = state;
    request.status = StreamRequestStatus.STREAMING;
  }

  _complete(request, result) {
    const provider = this.providers.get(request.provider_type);
    try {
      provider?.complete_stream(request, {
        result,
        system: this,
      });
    } catch (stream_error) {
      this._fail(request, stream_error);
      return;
    }

    request.result = result;
    request.error = null;
    request.status = StreamRequestStatus.COMPLETED;
    this._remove_request(request);
  }

  _fail(request, stream_error) {
    if (is_terminal_status(request.status)) {
      return;
    }

    const error =
      stream_error instanceof Error
        ? stream_error
        : new Error(String(stream_error ?? "Streaming request failed"));
    const provider = this.providers.get(request.provider_type);
    try {
      provider?.cancel_stream(request, {
        reason: error.message,
        error,
        system: this,
      });
    } catch {
      // Preserve the error that caused the request to fail.
    }

    request.result = null;
    request.error = error;
    request.status = StreamRequestStatus.FAILED;
    this._remove_request(request);
  }

  _remove_request(request) {
    this.requests.delete(request.id);
    request._resolve_finished?.(request);
    request._resolve_finished = null;
  }

  static get() {
    if (!this.instance) {
      this.instance = new StreamingSystem();
    }
    return this.instance;
  }

  static install(simulation_core) {
    const instance = this.get();
    if (!simulation_core?.register_simulation_layer) {
      throw new Error("StreamingSystem.install requires a simulation core.");
    }
    if (!simulation_core.simulation_layers?.includes(instance)) {
      simulation_core.register_simulation_layer(instance);
    }
    return instance;
  }

  static register_provider(provider_or_class, options = {}) {
    return this.get().register_provider(provider_or_class, options);
  }

  static stream(provider_type, target, options = {}) {
    return this.get().stream(provider_type, target, options);
  }

  static cancel_target(provider_type, target, reason) {
    return this.get().cancel_target(provider_type, target, reason);
  }

  static serialize(provider_type, value, context = {}) {
    return this.get().serialize(provider_type, value, context);
  }

  static deserialize(provider_type, payload, context = {}) {
    return this.get().deserialize(provider_type, payload, context);
  }
}
