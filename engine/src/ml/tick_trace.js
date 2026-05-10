const MAX_TRACE_LINES = 1000;

function now_ms() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function format_shape(value) {
  if (value === null || value === undefined) {
    return "null";
  }
  if (Array.isArray(value)) {
    return `[${value.join(",")}]`;
  }
  return String(value);
}

function format_details(details) {
  if (!details) {
    return "";
  }

  if (typeof details === "string") {
    return details;
  }

  const parts = [];
  for (const [key, value] of Object.entries(details)) {
    parts.push(`${key}=${format_shape(value)}`);
  }

  return parts.join(" ");
}

class MLTraceController {
  enabled = false;
  frame_limit = Infinity;
  frames_remaining = Infinity;
  tick_index = 0;
  lines = [];

  enable(frame_count = Infinity) {
    this.enabled = true;
    this.frame_limit = frame_count;
    this.frames_remaining = frame_count;
    this.log("trace.enabled", Number.isFinite(frame_count) ? { frames: frame_count } : "");
  }

  disable() {
    this.log("trace.disabled");
    this.enabled = false;
    this.frame_limit = Infinity;
    this.frames_remaining = Infinity;
  }

  clear() {
    this.lines.length = 0;
  }

  begin_tick(delta_time, subnet_count) {
    if (!this.enabled) {
      return false;
    }

    if (this.frames_remaining <= 0) {
      this.disable();
      return false;
    }

    this.tick_index++;
    this.log("tick.begin", { tick: this.tick_index, delta: delta_time, subnets: subnet_count });

    if (Number.isFinite(this.frames_remaining)) {
      this.frames_remaining--;
    }

    return true;
  }

  log(event, details = "") {
    if (!this.enabled) {
      return;
    }

    const suffix = format_details(details);
    const line = `[${now_ms().toFixed(2)}] ${event}${suffix ? ` ${suffix}` : ""}`;
    this.lines.push(line);

    if (this.lines.length > MAX_TRACE_LINES) {
      this.lines.splice(0, this.lines.length - MAX_TRACE_LINES);
    }

  }

  get_lines() {
    return this.lines;
  }

  get_state() {
    return {
      enabled: this.enabled,
      frame_limit: this.frame_limit,
      frames_remaining: this.frames_remaining,
      tick_index: this.tick_index,
      line_count: this.lines.length,
    };
  }
}

export const MLTrace = new MLTraceController();
