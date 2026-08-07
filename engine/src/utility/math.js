const EPSILON = 0.000001;

/**
 * Enum representing different easing types for animations.
 * @enum {string}
 */
export const EasingType = {
  Linear: 0,
  EaseInQuad: 1,
  EaseOutQuad: 2,
  EaseInOutQuad: 3,
  EaseInCubic: 4,
  EaseOutCubic: 5,
  EaseInOutCubic: 6,
  EaseInQuart: 7,
  EaseOutQuart: 8,
  EaseInOutQuart: 9,
  EaseInQuint: 10,
  EaseOutQuint: 11,
  EaseInOutQuint: 12,
  EaseInSine: 13,
  EaseOutSine: 14,
  EaseInOutSine: 15,
  EaseInExpo: 16,
  EaseOutExpo: 17,
  EaseInOutExpo: 18,
  EaseInCirc: 19,
  EaseOutCirc: 20,
  EaseInOutCirc: 21,
  EaseInElastic: 22,
  EaseOutElastic: 23,
  EaseInOutElastic: 24,
  EaseInBack: 25,
  EaseOutBack: 26,
  EaseInOutBack: 27,
  EaseInBounce: 28,
  EaseOutBounce: 29,
  EaseInOutBounce: 30,
};

export const Easing = {
  [EasingType.Linear]: (t) => t,
  [EasingType.EaseInQuad]: (t) => t * t,
  [EasingType.EaseOutQuad]: (t) => t * (2 - t),
  [EasingType.EaseInOutQuad]: (t) =>
    t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  [EasingType.EaseInCubic]: (t) => t * t * t,
  [EasingType.EaseOutCubic]: (t) => (t - 1) * (t - 1) * (t - 1) + 1,
  [EasingType.EaseInOutCubic]: (t) =>
    t < 0.5
      ? 4 * t * t * t
      : 1 - Math.pow(-2 * t + 2, 3) / 2,
  [EasingType.EaseInQuart]: (t) => t * t * t * t,
  [EasingType.EaseOutQuart]: (t) => 1 - (t - 1) * (t - 1) * (t - 1) * (t - 1),
  [EasingType.EaseInOutQuart]: (t) =>
    t < 0.5 ? 8 * t * t * t * t : 1 - 8 * (t - 1) * (t - 1) * (t - 1) * (t - 1),
  [EasingType.EaseInQuint]: (t) => t * t * t * t * t,
  [EasingType.EaseOutQuint]: (t) =>
    1 + (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1),
  [EasingType.EaseInOutQuint]: (t) =>
    t < 0.5
      ? 16 * t * t * t * t * t
      : 1 + 16 * (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1),
  [EasingType.EaseInSine]: (t) => 1 - Math.cos((t * Math.PI) / 2),
  [EasingType.EaseOutSine]: (t) => Math.sin((t * Math.PI) / 2),
  [EasingType.EaseInOutSine]: (t) => -0.5 * (Math.cos(Math.PI * t) - 1),
  [EasingType.EaseInExpo]: (t) => (t === 0 ? 0 : Math.pow(2, 10 * (t - 1))),
  [EasingType.EaseOutExpo]: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  [EasingType.EaseInOutExpo]: (t) =>
    t === 0
      ? 0
      : t === 1
      ? 1
      : t < 0.5
      ? Math.pow(2, 20 * t - 10) / 2
      : (2 - Math.pow(2, -20 * t + 10)) / 2,
  [EasingType.EaseInCirc]: (t) => 1 - Math.sqrt(1 - t * t),
  [EasingType.EaseOutCirc]: (t) => Math.sqrt(t * t - t + 1),
  [EasingType.EaseInOutCirc]: (t) =>
    t < 0.5
      ? (1 - Math.sqrt(1 - 2 * t * (2 * t))) / 2
      : (Math.sqrt(t * t - t + 1) + 1) / 2,
  [EasingType.EaseInElastic]: (t) =>
    t === 0
      ? 0
      : t === 1
      ? 1
      : -Math.pow(2, 10 * (t - 1)) *
        Math.sin(((t - 1.1) * (2 * Math.PI)) / 0.4),
  [EasingType.EaseOutElastic]: (t) =>
    t === 0
      ? 0
      : t === 1
      ? 1
      : Math.pow(2, -10 * t) * Math.sin(((t - 0.9) * (2 * Math.PI)) / 0.4) + 1,
  [EasingType.EaseInOutElastic]: (t) =>
    t === 0
      ? 0
      : t === 1
      ? 1
      : t < 0.5
      ? -0.5 *
        (Math.pow(2, 20 * t - 10) *
          Math.sin(((20 * t - 11.12) * (2 * Math.PI)) / 0.4))
      : 0.5 *
          (Math.pow(2, -20 * t + 10) *
            Math.sin(((20 * t - 11.12) * (2 * Math.PI)) / 0.4)) +
        1,
  [EasingType.EaseInBack]: (t) => {
    const c1 = 1.70158;
    return c1 * t * t * t - c1 * t * t;
  },
  [EasingType.EaseOutBack]: (t) => {
    const c1 = 1.70158;
    return 1 + c1 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  [EasingType.EaseInOutBack]: (t) => {
    const c2 = 2.5949095;
    return t < 0.5
      ? (Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
      : (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
  },
  [EasingType.EaseInBounce]: (t) => 1 - Easing[EasingType.EaseOutBounce](1 - t),
  [EasingType.EaseOutBounce]: (t) => {
    const n1 = 7.5625;
    const d1 = 2.75;
    if (t < 1 / d1) {
      return n1 * t * t;
    } else if (t < 2 / d1) {
      return n1 * (t -= 1.5 / d1) * t + 0.75;
    } else if (t < 2.5 / d1) {
      return n1 * (t -= 2.25 / d1) * t + 0.9375;
    } else {
      return n1 * (t -= 2.625 / d1) * t + 0.984375;
    }
  },
  [EasingType.EaseInOutBounce]: (t) =>
    t < 0.5
      ? (1 - Easing[EasingType.EaseOutBounce](1 - 2 * t)) / 2
      : (1 + Easing[EasingType.EaseOutBounce](2 * t - 1)) / 2,
};

export function radians(deg) {
  return deg * (Math.PI / 180);
}

export function degrees(rad) {
  return rad * (180 / Math.PI);
}

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function ispot(value) {
  return value > 0 && (value & (value - 1)) === 0;
}

export function npot(value) {
  value = Math.floor(value);
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return ++value >>> 0;
}

export function ppot(value) {
  value = Math.floor(value);
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return (value - (value >> 1)) >>> 0;
}

export function halton(index, base) {
  let result = 0.0;
  let f = 1.0;
  while (index > 0) {
    f /= base;
    result += f * (index % base);
    index = Math.floor(index / base);
  }
  return result;
}

export function near_zero(value) {
  return Math.abs(value) < 1e-6;
}

export function near_equal(a, b) {
  return Math.abs(a - b) < 1e-6;
}

export function vec_near_zero(v) {
  return near_zero(v[0]) && near_zero(v[1]) && near_zero(v[2]);
}

export function vec_near_equal(a, b) {
  return near_equal(a[0], b[0]) && near_equal(a[1], b[1]) && near_equal(a[2], b[2]);
}

export function bytes_to_mb(bytes) {
  return (bytes / (1024.0 * 1024.0)).toFixed(2);
}

export function clamp_unit(value) {
  return Math.max(-1.0, Math.min(1.0, Number.isFinite(value) ? value : 0.0));
}

export function encode_snorm8(value) {
  const scaled = Math.round(clamp_unit(value) * 127.0);
  return scaled & 0xff;
}

export function pack_snorm4x8(x, y, z, w = 0.0) {
  return (
    encode_snorm8(x) |
    (encode_snorm8(y) << 8) |
    (encode_snorm8(z) << 16) |
    (encode_snorm8(w) << 24)
  ) >>> 0;
}

export function ceil_div(a, b) {
  return Math.ceil(a / b);
}

export function floor_div(a, b) {
  return Math.floor(a / b);
}

export function floor_to_multiple(value, multiple) {
  return Math.floor(value / multiple) * multiple;
}

/**
 * Returns an euler angle representation of a quaternion, in degrees
 * @param  {vec3} out Euler angles, pitch-yaw-roll
 * @param  {quat} mat Quaternion
 * @return {vec3} out
 */
export function quat_to_euler(out, q) {
  // Assuming q is in the form [x, y, z, w]
  const [x, y, z, w] = q;

  // Calculate pitch (x-axis rotation)
  const sinp = 2.0 * (w * y - z * x);
  if (Math.abs(sinp) >= 1) {
    out[1] = Math.copySign(Math.PI / 2, sinp); // use 90 degrees if out of range
  } else {
    out[0] = Math.asin(sinp);
  }

  // Calculate yaw (y-axis rotation)
  const siny_cosp = 2.0 * (w * z + x * y);
  const cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  out[1] = Math.atan2(siny_cosp, cosy_cosp);

  // Calculate roll (z-axis rotation)
  const sinr_cosp = 2.0 * (w * x + y * z);
  const cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  out[2] = Math.atan2(sinr_cosp, cosr_cosp);

  // Convert to degrees
  out[0] *= (180 / Math.PI);
  out[1] *= (180 / Math.PI);
  out[2] *= (180 / Math.PI);

  return out; // [pitch, yaw, roll] in degrees
}

export function direction_vector_to_quat(vec) {
    // This assumes vec is normalized
    const up = [0, 1, 0];
    const right = [0, 0, 0];
    const forward = [-vec[0], -vec[1], -vec[2]];
    
    // Calculate right vector using cross product of up and forward
    right[0] = up[1] * forward[2] - up[2] * forward[1];
    right[1] = up[2] * forward[0] - up[0] * forward[2];
    right[2] = up[0] * forward[1] - up[1] * forward[0];
    
    // Normalize right vector
    const right_length = Math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right[0] /= right_length;
    right[1] /= right_length;
    right[2] /= right_length;
    
    // Recalculate up vector to ensure orthogonality
    up[0] = forward[1] * right[2] - forward[2] * right[1];
    up[1] = forward[2] * right[0] - forward[0] * right[2];
    up[2] = forward[0] * right[1] - forward[1] * right[0];
    
    // Convert to quaternion (using rotation matrix to quaternion conversion)
    const trace = right[0] + up[1] + forward[2];
    let qw, qx, qy, qz;
    
    if (trace > 0) {
        const S = Math.sqrt(trace + 1.0) * 2;
        qw = 0.25 * S;
        qx = (up[2] - forward[1]) / S;
        qy = (forward[0] - right[2]) / S;
        qz = (right[1] - up[0]) / S;
    } else if (right[0] > up[1] && right[0] > forward[2]) {
        const S = Math.sqrt(1.0 + right[0] - up[1] - forward[2]) * 2;
        qw = (up[2] - forward[1]) / S;
        qx = 0.25 * S;
        qy = (right[1] + up[0]) / S;
        qz = (right[2] + forward[0]) / S;
    } else if (up[1] > forward[2]) {
        const S = Math.sqrt(1.0 + up[1] - right[0] - forward[2]) * 2;
        qw = (forward[0] - right[2]) / S;
        qx = (right[1] + up[0]) / S;
        qy = 0.25 * S;
        qz = (up[2] + forward[1]) / S;
    } else {
        const S = Math.sqrt(1.0 + forward[2] - right[0] - up[1]) * 2;
        qw = (right[1] - up[0]) / S;
        qx = (right[2] + forward[0]) / S;
        qy = (up[2] + forward[1]) / S;
        qz = 0.25 * S;
    }
    
    return [qx, qy, qz, qw];
}

export function is_vec_nearly_zero(vec, epsilon = EPSILON) {
  return Math.abs(vec[0]) < epsilon && Math.abs(vec[1]) < epsilon && Math.abs(vec[2]) < epsilon;
}
