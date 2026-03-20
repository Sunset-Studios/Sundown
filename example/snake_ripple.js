const DEFAULT_PULSE_DURATION = 0.28;
const DEFAULT_SEGMENT_DELAY = 0.06;
const DEFAULT_AMPLITUDE = 0.28;
const DEFAULT_MAX_MULTIPLIER = 1.45;

export function createRippleBurst(options = {}) {
  return {
    elapsed: options.elapsed ?? 0,
    pulseDuration: options.pulseDuration ?? DEFAULT_PULSE_DURATION,
    segmentDelay: options.segmentDelay ?? DEFAULT_SEGMENT_DELAY,
    amplitude: options.amplitude ?? DEFAULT_AMPLITUDE,
    maxMultiplier: options.maxMultiplier ?? DEFAULT_MAX_MULTIPLIER,
  };
}

export function advanceRippleBursts(bursts, deltaTime, segmentCount) {
  const clampedDelta = Number.isFinite(deltaTime) ? Math.max(0, deltaTime) : 0;
  const maxIndex = Math.max(0, segmentCount - 1);

  return bursts
    .map((burst) => ({
      ...burst,
      elapsed: burst.elapsed + clampedDelta,
    }))
    .filter(
      (burst) => burst.elapsed <= burst.pulseDuration + burst.segmentDelay * maxIndex
    );
}

export function getRippleScaleMultiplier(segmentIndex, bursts) {
  if (!Array.isArray(bursts) || bursts.length === 0) {
    return 1;
  }

  let totalOffset = 0;
  let maxMultiplier = DEFAULT_MAX_MULTIPLIER;

  for (const burst of bursts) {
    maxMultiplier = Math.max(maxMultiplier, burst.maxMultiplier ?? DEFAULT_MAX_MULTIPLIER);

    const segmentDelay = burst.segmentDelay ?? DEFAULT_SEGMENT_DELAY;
    const pulseDuration = burst.pulseDuration ?? DEFAULT_PULSE_DURATION;
    const localTime = (burst.elapsed ?? 0) - segmentIndex * segmentDelay;

    if (localTime < 0 || localTime > pulseDuration) {
      continue;
    }

    const phase = localTime / pulseDuration;
    totalOffset += Math.sin(Math.PI * phase) * (burst.amplitude ?? DEFAULT_AMPLITUDE);
  }

  return Math.min(maxMultiplier, 1 + totalOffset);
}
