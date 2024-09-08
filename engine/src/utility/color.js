/**
 * Generates a random RGB color.
 * @returns {Object} An object with r, g, and b properties, each ranging from 0 to 255.
 */
export function random_rgb() {
  return {
    r: Math.floor(Math.random() * 256),
    g: Math.floor(Math.random() * 256),
    b: Math.floor(Math.random() * 256)
  };
}
