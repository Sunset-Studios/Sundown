export class RollingAverage {
  capacity = 30;
  values = null;
  sum = 0;
  index = 0;
  count = 0;

  constructor(capacity = 30) {
    this.capacity = Math.max(1, capacity);
    this.values = new Float64Array(this.capacity);
    this.sum = 0;
    this.index = 0;
    this.count = 0;
  }

  add_sample(value) {
    const clamped_value = Number.isFinite(value) ? value : 0;
    if (this.count < this.capacity) {
      this.values[this.index] = clamped_value;
      this.sum += clamped_value;
      this.count += 1;
    } else {
      this.sum -= this.values[this.index];
      this.values[this.index] = clamped_value;
      this.sum += clamped_value;
    }
    this.index = (this.index + 1) % this.capacity;
  }

  get_average() {
    if (this.count === 0) return 0;
    return this.sum / this.count;
  }

  reset() {
    this.sum = 0;
    this.index = 0;
    this.count = 0;
    this.values.fill(0);
  }
}


