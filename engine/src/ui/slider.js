export const SliderMode = Object.freeze({
  Bar: "bar",
  Numeric: "numeric",
});

const default_min = 0;
const default_max = 1;
const default_scrub_pixels = 100;
const max_rounding_precision = 12;

function finite_number(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function decimal_places(value) {
  const string_value = String(value).toLowerCase();
  if (string_value.includes("e-")) {
    const [coefficient, exponent] = string_value.split("e-");
    const coefficient_places = (coefficient.split(".")[1] || "").length;
    return Math.min(max_rounding_precision, finite_number(exponent, 0) + coefficient_places);
  }

  return Math.min(max_rounding_precision, (string_value.split(".")[1] || "").length);
}

/**
 * Resolves the numeric portion of a slider config and clamps its incoming value.
 * Reversed bounds are accepted and normalized to an ascending range.
 */
export function resolve_slider_value(value, config = {}) {
  let min = finite_number(config.min, default_min);
  let max = finite_number(config.max, default_max);
  if (max < min) {
    [min, max] = [max, min];
  }

  const configured_step = finite_number(config.step, 0);
  const step = configured_step > 0 ? configured_step : 0;
  const fallback_value = min;
  const source_value = Number(value);
  const finite_source_value = finite_number(source_value, fallback_value);
  const clamped_value = Math.min(max, Math.max(min, finite_source_value));
  const resolved_value =
    step > 0 ? quantize_slider_value(clamped_value, min, max, step) : clamped_value;

  return {
    min,
    max,
    step,
    range: max - min,
    value: resolved_value,
    source_value,
  };
}

/**
 * Snaps a value to a step anchored at min, then clamps it to the slider range.
 */
export function quantize_slider_value(value, min, max, step) {
  const clamped_value = Math.min(max, Math.max(min, finite_number(value, min)));
  if (!(step > 0) || max <= min) {
    return clamped_value;
  }
  if (clamped_value === min || clamped_value === max) {
    return clamped_value;
  }

  const snapped_value = min + Math.round((clamped_value - min) / step) * step;
  const precision = Math.max(decimal_places(min), decimal_places(max), decimal_places(step));
  const rounded_value = Number(snapped_value.toFixed(precision));
  return Math.min(max, Math.max(min, rounded_value));
}

export function slider_ratio(value, resolved) {
  if (!(resolved.range > 0)) {
    return 0;
  }
  return Math.min(1, Math.max(0, (value - resolved.min) / resolved.range));
}

export function slider_value_from_ratio(ratio, resolved) {
  const unit_ratio = Math.min(1, Math.max(0, finite_number(ratio, 0)));
  const value = resolved.min + resolved.range * unit_ratio;
  return resolved.step > 0
    ? quantize_slider_value(value, resolved.min, resolved.max, resolved.step)
    : Math.min(resolved.max, Math.max(resolved.min, value));
}

export function slider_value_from_scrub(start_value, delta_x, resolved, config = {}) {
  const default_speed =
    resolved.step > 0
      ? resolved.step
      : resolved.range / finite_number(config.scrub_pixels, default_scrub_pixels);
  const scrub_speed = Math.abs(finite_number(config.scrub_speed, default_speed));
  const value = start_value + finite_number(delta_x, 0) * scrub_speed;
  return resolved.step > 0
    ? quantize_slider_value(value, resolved.min, resolved.max, resolved.step)
    : Math.min(resolved.max, Math.max(resolved.min, value));
}

export function format_slider_value(value, resolved, config = {}) {
  const formatter = config.format_value ?? config.format;
  if (typeof formatter === "function") {
    return String(formatter(value));
  }

  const configured_precision = Number(config.precision);
  const precision = Number.isInteger(configured_precision)
    ? Math.min(max_rounding_precision, Math.max(0, configured_precision))
    : resolved.step > 0
      ? decimal_places(resolved.step)
      : 3;
  const numeric_text = Number(value.toFixed(precision)).toString();
  return `${config.prefix ?? ""}${numeric_text}${config.suffix ?? ""}`;
}

function pointer_ratio(pointer, axis_start, axis_end) {
  const axis_x = axis_end.x - axis_start.x;
  const axis_y = axis_end.y - axis_start.y;
  const length_squared = axis_x * axis_x + axis_y * axis_y;
  if (!(length_squared > 0)) {
    return 0;
  }

  const pointer_x = finite_number(pointer.x, axis_start.x) - axis_start.x;
  const pointer_y = finite_number(pointer.y, axis_start.y) - axis_start.y;
  return (pointer_x * axis_x + pointer_y * axis_y) / length_squared;
}

/**
 * Updates a slider from pointer input while preserving capture in the supplied state.
 * The state object is intentionally renderer-agnostic and may be shared by 2D and 3D.
 */
export function update_slider_interaction({
  state,
  widget_id,
  mode,
  value,
  resolved,
  config = {},
  pointer,
  hovered,
  axis_start,
  axis_end,
}) {
  let next_value = value;
  let activated = false;
  let released = false;
  const is_active = state.active_widget_id === widget_id;

  if (!pointer.pressed) {
    if (is_active) {
      state.active_widget_id = null;
      released = true;
    }
  } else if (state.active_widget_id === null && hovered && (pointer.clicked ?? pointer.pressed)) {
    state.active_widget_id = widget_id;
    state.drag_start_x = finite_number(pointer.x, 0);
    state.drag_start_value = value;
    activated = true;
  }

  const active = state.active_widget_id === widget_id;
  if (active && pointer.pressed) {
    if (mode === SliderMode.Numeric) {
      next_value = slider_value_from_scrub(
        state.drag_start_value,
        finite_number(pointer.x, 0) - state.drag_start_x,
        resolved,
        config
      );
    } else {
      next_value = slider_value_from_ratio(pointer_ratio(pointer, axis_start, axis_end), resolved);
    }
  }

  return {
    value: next_value,
    changed: !Object.is(next_value, resolved.source_value),
    active,
    activated,
    released,
  };
}
