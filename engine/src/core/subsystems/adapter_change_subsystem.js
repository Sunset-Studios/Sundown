import { Renderer } from "../../renderer/renderer.js";
import { SimulationLayer } from "../simulation_layer.js";
import { log, warn } from "../../utility/logging.js";

const default_check_interval_ms = 1000.0;

function collect_limit_snapshot(adapter) {
  if (!adapter || !adapter.limits) {
    return {};
  }

  const limit_keys = [
    "maxBindGroups",
    "maxBufferSize",
    "maxStorageBuffersPerShaderStage",
    "maxStorageBufferBindingSize",
    "maxTextureArrayLayers",
    "maxComputeWorkgroupStorageSize",
  ];

  const limit_snapshot = {};
  for (const key of limit_keys) {
    if (key in adapter.limits) {
      limit_snapshot[key] = adapter.limits[key];
    }
  }

  return limit_snapshot;
}

async function collect_adapter_snapshot(adapter) {
  if (!adapter) {
    return null;
  }

  let adapter_info = {};
  if (adapter.info) {
    adapter_info = adapter.info;
  } else if (typeof adapter.requestAdapterInfo === "function") {
    try {
      adapter_info = await adapter.requestAdapterInfo();
    } catch (_error) {
      adapter_info = {};
    }
  }

  const features = [...adapter.features].sort();

  return {
    vendor: adapter_info.vendor || "unknown",
    architecture: adapter_info.architecture || "unknown",
    device: adapter_info.device || "unknown",
    description: adapter_info.description || "unknown",
    is_fallback_adapter: Boolean(adapter.isFallbackAdapter),
    features,
    limits: collect_limit_snapshot(adapter),
  };
}

function get_change_report(previous_snapshot, current_snapshot) {
  if (!previous_snapshot) {
    return null;
  }

  const changed_keys = [];

  const scalar_keys = ["vendor", "architecture", "device", "description", "is_fallback_adapter"];
  for (const key of scalar_keys) {
    if (previous_snapshot[key] !== current_snapshot[key]) {
      changed_keys.push(key);
    }
  }

  if (JSON.stringify(previous_snapshot.features) !== JSON.stringify(current_snapshot.features)) {
    changed_keys.push("features");
  }

  if (JSON.stringify(previous_snapshot.limits) !== JSON.stringify(current_snapshot.limits)) {
    changed_keys.push("limits");
  }

  if (!changed_keys.length) {
    return null;
  }

  return {
    changed_keys,
    previous_snapshot,
    current_snapshot,
  };
}

export class AdapterChangeSubsystem extends SimulationLayer {
  check_interval_ms = default_check_interval_ms;
  elapsed_ms = 0;
  is_check_in_flight = false;
  last_snapshot = null;
  power_preference = "high-performance";

  async init() {
    const renderer = Renderer.get();
    this.last_snapshot = await collect_adapter_snapshot(renderer?.adapter ?? null);
    this.elapsed_ms = 0;
  }

  update(delta_time) {
    super.update(delta_time);

    if (this.is_check_in_flight) {
      return;
    }

    this.elapsed_ms += delta_time * 1000;

    if (this.elapsed_ms < this.check_interval_ms) {
      return;
    }

    this.elapsed_ms = 0;
    this.is_check_in_flight = true;

    //log('[AdapterChangeSubsystem] Checking for adapter change...');

    const that = this;
    this.check_for_adapter_change().finally(() => {
      that.is_check_in_flight = false;
    });
  }

  async check_for_adapter_change() {
    if (!navigator.gpu) {
      return;
    }

    let current_adapter = null;
    try {
      current_adapter = await navigator.gpu.requestAdapter({
        powerPreference: this.power_preference,
      });
    } catch (request_error) {
      warn("[AdapterChangeSubsystem] Adapter check failed", request_error);
      return;
    }

    if (!current_adapter) {
      warn("[AdapterChangeSubsystem] Unable to request adapter while monitoring");
      return;
    }

    // log(`[AdapterChangeSubsystem] Adapter check completed:`);
    // console.log(current_adapter);

    const current_snapshot = await collect_adapter_snapshot(current_adapter);
    const change_report = get_change_report(this.last_snapshot, current_snapshot);
    this.last_snapshot = current_snapshot;

    if (change_report) {
      log("[AdapterChangeSubsystem] Adapter info changed", change_report);
    }
  }
}
