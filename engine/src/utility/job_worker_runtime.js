const job_handlers = new Map();
const running_jobs = new Map();

export function register_job_handler(type, handler) {
  job_handlers.set(type, handler);
}

export function create_job_result(payload, transferables = []) {
  return {
    payload,
    transferables,
  };
}

export function resolve_href(path, base_href = self.location.href) {
  return new URL(path, base_href).href;
}

export function is_html_content_type(content_type = "") {
  return content_type.includes("text/html") || content_type.includes("application/xhtml+xml");
}

export function start_job_worker() {
  self.onmessage = async (event) => {
    const message = event.data;
    if (!message?.kind) {
      return;
    }

    if (message.kind === "cancel") {
      const controller = running_jobs.get(message.job_id);
      controller?.abort();
      return;
    }

    if (message.kind !== "submit") {
      return;
    }

    const handler = job_handlers.get(message.type);
    if (!handler) {
      self.postMessage({
        kind: "error",
        job_id: message.job_id,
        error: `No worker job handler registered for '${message.type}'`,
      });
      return;
    }

    const controller = new AbortController();
    running_jobs.set(message.job_id, controller);

    try {
      const result = await handler(message.payload, {
        signal: controller.signal,
        post_progress(progress) {
          self.postMessage({
            kind: "progress",
            job_id: message.job_id,
            progress,
          });
        },
      });

      if (controller.signal.aborted) {
        self.postMessage({
          kind: "cancelled",
          job_id: message.job_id,
        });
        return;
      }

      const payload = result?.payload ?? result ?? null;
      const transferables = result?.transferables ?? [];
      self.postMessage(
        {
          kind: "complete",
          job_id: message.job_id,
          payload,
        },
        transferables
      );
    } catch (error) {
      if (controller.signal.aborted) {
        self.postMessage({
          kind: "cancelled",
          job_id: message.job_id,
        });
      } else {
        self.postMessage({
          kind: "error",
          job_id: message.job_id,
          error: error?.message ?? String(error),
        });
      }
    } finally {
      running_jobs.delete(message.job_id);
    }
  };
}
