export class Fragment {
  static resize_array(
    obj,
    key,
    new_size,
    ArrayType = Float32Array,
    stride = 1,
    wipe = false
  ) {
    if (obj[key].length < new_size * stride) {
      const prev = obj[key];
      obj[key] = new ArrayType(new_size * stride);
      if (wipe) {
        obj[key].fill(0);
      } else {
        obj[key].set(prev);
      }
    }
  }
}
