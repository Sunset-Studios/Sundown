import JobWorker from "./job_worker.js?worker";
import { FrameQueueAllocator, RandomAccessAllocator } from "../memory/allocator.js";
import { SimulationLayer } from "../core/simulation_layer.js";
import SimulationCore from "../core/simulation_core.js";

const undefined_string = "undefined";
const function_string = "function";
const could_not_deserialize_error_string = "Worker job message could not be deserialized";
const worker_job_system_not_supported_error_string = "Worker job system is not supported in this environment";
const worker_job_failed_error_string = "Worker job failed";
const job_cancelled_error_string = "Job cancelled";
const job_system_shutdown_error_string = "Job system shutdown";
const submit_message_kind_string = "submit";
const progress_message_kind_string = "progress";
const complete_message_kind_string = "complete";
const error_message_kind_string = "error";
const cancelled_message_kind_string = "cancelled";
const cancel_message_kind_string = "cancel";

export const JobStatus = {
  NONE: 0,
  PENDING: 1,
  RUNNING: 2,
  COMPLETED: 3,
  FAILED: 4,
  CANCELLED: 5,
};

class WorkerJob {
  job_id = null;
  type = null;
  payload = null;
  transferables = [];
  status = JobStatus.NONE;
}

class WorkerSlot {
  worker = null;
  busy = false;
  current_job_id = null;
}

class WorkerJobHandle {
  job_system = null;
  job_id = null;
  progress_listeners = new Set();
  promise = null;

  constructor(job_system, job_id) {
    this.job_system = job_system;
    this.job_id = job_id;

    this.promise = new Promise((resolve, reject) => {
      this._resolve = resolve;
      this._reject = reject;
    });
  }

  on_progress(listener) {
    if (typeof listener === function_string) {
      this.progress_listeners.add(listener);
    }
    return () => {
      this.progress_listeners.delete(listener);
    };
  }

  emit_progress(progress) {
    for (const listener of this.progress_listeners) {
      if (typeof listener === function_string) {
        listener(progress);
      }
    }
  }

  resolve(result) {
    this._resolve(result);
  }

  reject(error) {
    this._reject(error);
  }

  cancel() {
    this.job_system.cancel(this.job_id);
  }
}

export class JobSystem extends SimulationLayer {
  static instance = null;
  static next_job_id = 1;

  worker_count = 1;
  handles = new Map();
  queue = null;
  workers = null;
  round_robin_index = 0;

  constructor() {
    super();
    this.name = "JobSystem";
  }

  init() {
    super.init();

    if (JobSystem.is_supported()) {
      const hardware_concurrency =
        typeof navigator !== undefined_string && navigator.hardwareConcurrency
          ? navigator.hardwareConcurrency
          : 1;
      this.worker_count = Math.max(1, hardware_concurrency - 1);
    }

    this.queue = new FrameQueueAllocator(1024, WorkerJob);
    this.workers = new RandomAccessAllocator(this.worker_count, WorkerSlot);

    for (let i = 0; i < this.worker_count; i++) {
      const worker = new JobWorker();

      worker.onmessage = (event) => {
        this.#handle_worker_message(slot, event.data);
      };
      worker.onerror = (event) => {
        this.#handle_worker_fail(slot, this.#normalize_worker_error(event));
      };
      worker.onmessageerror = () => {
        this.#handle_worker_fail(slot, new Error(could_not_deserialize_error_string));
      };

      const slot = this.workers.allocate();
      slot.worker = worker;
      slot.busy = false;
      slot.current_job_id = null;
    }
  }

  update(delta_time) {
    super.update(delta_time);
    this.#schedule();
  }

  cleanup() {
    this.terminate();
    super.cleanup();
  }

  submit(type, payload = null, options = {}) {
    const job_id = JobSystem.next_job_id++;
    const handle = new WorkerJobHandle(this, job_id);

    this.handles.set(job_id, handle);

    const job = this.queue.enqueue();
    job.job_id = job_id;
    job.type = type;
    job.payload = payload;
    job.transferables = options.transferables ?? [];
    job.status = JobStatus.PENDING;

    return handle;
  }

  cancel(job_id) {
    for (let i = 0; i < this.queue.length; i++) {
      const job = this.queue.get(i);
      if (job.job_id === job_id) {
        const handle = this.handles.get(job.job_id);
        if (handle) {
          handle.reject(new Error(job_cancelled_error_string));
          this.handles.delete(job.job_id);
        }
        job.status = JobStatus.CANCELLED;
      }
    }

    for (let i = 0; i < this.workers.length; i++) {
      const slot = this.workers.get(i);
      if (slot.current_job_id === job_id) {
        slot.worker.postMessage({
          kind: cancel_message_kind_string,
          job_id,
        });
      }
    }
  }

  terminate() {
    const shutdown_error = new Error(job_system_shutdown_error_string);

    for (const handle of this.handles.values()) {
      handle.reject(shutdown_error);
    }
    this.handles.clear();

    for (let i = 0; i < this.workers.length; i++) {
      const slot = this.workers.get(i);
      slot.worker?.terminate();
      slot.worker = null;
      slot.busy = false;
      slot.current_job_id = null;
    }

    this.queue.reset();
    this.workers.reset();
    this.round_robin_index = 0;
  }

  #schedule() {
    if (this.queue.length === 0 || this.workers.length === 0) {
      return;
    }

    for (let attempt = 0; attempt < this.workers.length; attempt++) {
      const slot = this.workers.get(this.round_robin_index);
      this.round_robin_index = (this.round_robin_index + 1) % this.workers.length;

      if (slot.busy) {
        continue;
      }

      const next_job = this.queue.dequeue();
      if (!next_job) {
        return;
      }

      if (next_job.status !== JobStatus.PENDING) {
        attempt--;
        continue;
      }

      slot.busy = true;
      slot.current_job_id = next_job.job_id;

      try {
        slot.worker.postMessage(
          {
            kind: submit_message_kind_string,
            job_id: next_job.job_id,
            type: next_job.type,
            payload: next_job.payload,
          },
          next_job.transferables
        );
      } catch (error) {
        this.#handle_worker_fail(slot, error);
      }

      if (this.queue.length === 0) {
        return;
      }
    }
  }

  #handle_worker_message(slot, message) {
    if (!message?.kind) {
      return;
    }

    const handle = this.handles.get(message.job_id);

    switch (message.kind) {
      case progress_message_kind_string:
        handle?.emit_progress(message.progress);
        break;
      case complete_message_kind_string:
        this.#handle_worker_success(slot, message.payload);
        break;
      case error_message_kind_string:
        this.#handle_worker_fail(slot, new Error(message.error ?? worker_job_failed_error_string));
        break;
      case cancelled_message_kind_string:
        this.#handle_worker_fail(slot, new Error(job_cancelled_error_string));
        break;
    }
  }

  #handle_worker_success(slot, payload) {
    const job_id = slot.current_job_id;
    slot.busy = false;
    slot.current_job_id = null;

    const handle = job_id !== null ? this.handles.get(job_id) : null;
    if (handle) {
      handle.resolve(payload);
      this.handles.delete(job_id);
    }
  }

  #handle_worker_fail(slot, error) {
    const job_id = slot.current_job_id;
    slot.busy = false;
    slot.current_job_id = null;

    if (job_id !== null) {
      const handle = this.handles.get(job_id);
      if (handle) {
        handle.reject(error);
        this.handles.delete(job_id);
      }
    }
  }

  #normalize_worker_error(event) {
    if (event instanceof Error) {
      return event;
    }

    const parts = [];
    if (event?.message) {
      parts.push(event.message);
    }
    if (event?.filename) {
      const line = event?.lineno ?? 0;
      const column = event?.colno ?? 0;
      parts.push(`${event.filename}:${line}:${column}`);
    }
    if (parts.length === 0 && event?.type) {
      parts.push(`Worker emitted '${event.type}' during startup or execution`);
    }

    return new Error(parts.join(" ") || worker_job_failed_error_string);
  }

  static is_supported() {
    return typeof Worker !== undefined_string;
  }

  static get() {
    if (!this.instance) {
      this.instance = new JobSystem();
    }
    return this.instance;
  }

  static install() {
    const instance = this.get();
    if (!SimulationCore.simulation_layers.includes(instance)) {
      SimulationCore.register_simulation_layer(instance);
    }
    return instance;
  }

  static submit(type, payload = null, options = {}) {
    return this.get().submit(type, payload, options);
  }

  static cancel(handle_or_job_id) {
    this.get().cancel(handle_or_job_id);
  }
}
