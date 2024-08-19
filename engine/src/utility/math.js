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
    t < 0.5 ? 4 * t * t * t : (t - 1) * (t - 1) * (t - 1) + 1,
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
  [EasingType.EaseInBack]: (t) => t * t * (2.70158 * t - 1.70158),
  [EasingType.EaseOutBack]: (t) =>
    1 + (t - 1) * (t - 1) * (2.70158 * (t - 1) + 1.70158),
  [EasingType.EaseInOutBack]: (t) =>
    t < 0.5
      ? 0.5 * (t * t * (2.70158 * t - 1.70158))
      : 0.5 * (1 + (t - 1) * (t - 1) * (2.70158 * (t - 1) + 1.70158)),
  [EasingType.EaseInBounce]: (t) => 1 - Easing.EaseOutBounce(1 - t),
  [EasingType.EaseOutBounce]: (t) => 1 - Easing.EaseInBounce(t),
  [EasingType.EaseInOutBounce]: (t) =>
    t < 0.5
      ? 0.5 * (1 - Easing.EaseOutBounce(1 - 2 * t))
      : 0.5 * (1 + Easing.EaseOutBounce(2 * t - 1)),
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
