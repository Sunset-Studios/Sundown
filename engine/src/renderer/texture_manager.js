export class TextureManager {
  static max_texture_dimension = Number.POSITIVE_INFINITY;

  static set_max_texture_dimension(max_dimension) {
    if (max_dimension === Number.POSITIVE_INFINITY) {
      this.max_texture_dimension = max_dimension;
      return;
    }
    if (!Number.isFinite(max_dimension) || max_dimension < 1) {
      throw new Error("Texture max dimension must be a positive finite number or Infinity.");
    }
    this.max_texture_dimension = Math.max(1, Math.floor(max_dimension));
  }

  static get_max_texture_dimension() {
    return this.max_texture_dimension;
  }
}
