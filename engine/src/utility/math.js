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
  