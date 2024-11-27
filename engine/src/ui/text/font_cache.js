import { Renderer } from "../../renderer/renderer.js";
import { Font } from "./font.js";
import { read_file } from "../../utility/file_system.js";

const json_extension = ".json";

export class FontCache {
  static cache = new Map();
  static name_to_id = new Map();
  static next_id = 1;
  static font_paths = ['engine/fonts'];

  static register_font_path(path) {
    this.font_paths.push(path);
  }

  /**
   * Gets the font ID for a given font name
   * @param {string} font_name - The name of the font
   * @param {string|null} font_path - Optional path to font file
   * @returns {number|null} The font ID or null if font couldn't be loaded
   */
  static get_font(font_name, font_path = null) {
    // Return existing ID if font is already loaded
    if (this.name_to_id.has(font_name)) {
      return this.name_to_id.get(font_name);
    }

    const font = Font.create(Renderer.get().graphics_context, font_path);
    if (!font) return null;

    const font_id = this.next_id++;
    this.cache.set(font_id, font);
    this.name_to_id.set(font_name, font_id);
    return font_id;
  }

  /**
   * Gets the font object for a given font ID
   * @param {number} font_id - The ID of the font
   * @returns {Font|null} The font object or null if not found
   */
  static get_font_by_id(font_id) {
    return this.cache.get(font_id) || null;
  }

  /**
   * Gets the font ID for a given font name without loading
   * @param {string} font_name - The name of the font
   * @returns {number|null} The font ID or null if font isn't loaded
   */
  static get_font_id(font_name) {
    return this.name_to_id.get(font_name) || null;
  }

  static auto_load_fonts() {
    for (const path of this.font_paths) {
      this.scan_directory(path);
    }
  }

  static scan_directory(path) {
    const manifest = JSON.parse(read_file(`${path}/font_manifest.json`));

    if (!manifest) return;

    for (const entry of manifest.fonts) {
      if (this.is_valid_font_data_file(entry.path)) {
        this.get_font(entry.name, entry.path);
      }
    }
  }

  static is_valid_font_data_file(filepath) {
    return filepath.toLowerCase().endsWith(json_extension);
  }
}
