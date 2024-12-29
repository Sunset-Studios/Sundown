import { Renderer } from "../../renderer/renderer.js";
import { Font } from "./font.js";
import { read_file } from "../../utility/file_system.js";
import { Name } from "../../utility/names.js";

const json_extension = ".json";

export class FontCache {
  static cache = new Map();
  static font_paths = ['engine/fonts'];

  /**
   * Registers a new font path to be scanned for fonts.
   * @param {string} path - The path to the directory containing font files
   */
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
    const font_id = Name.from(font_name);
    if (this.cache.has(font_id)) {
      return font_id;
    }

    const font = Font.create(Renderer.get().graphics_context, font_path);
    if (!font) return null;

    this.cache.set(font_id, font);
    return font_id;
  }

  /**
   * Gets the font object for a given font ID
   * @param {number} font_id - The ID of the font
   * @returns {Font|null} The font object or null if not found
   */
  static get_font_object(font_id) {
    return this.cache.get(font_id) || null;
  }

  /**
   * Automatically loads all fonts from registered font paths by scanning their font manifests.
   * Iterates through each registered font path and calls scan_directory() to load fonts.
   */
  static auto_load_fonts() {
    for (const path of this.font_paths) {
      this.scan_directory(path);
    }
  }

  /**
   * Scans a directory for font manifests and loads all fonts found.
   * @param {string} path - The path to the directory to scan
   */
  static scan_directory(path) {
    const manifest = JSON.parse(read_file(`${path}/font_manifest.json`));

    if (!manifest) return;

    for (const entry of manifest.fonts) {
      if (this.is_valid_font_data_file(entry.path)) {
        this.get_font(entry.name, entry.path);
      }
    }
  }

  /**
   * Checks if a given file path is a valid font data file.
   * @param {string} filepath - The path to the file to check
   * @returns {boolean} True if the file is a valid font data file, false otherwise
   */
  static is_valid_font_data_file(filepath) {
    return filepath.toLowerCase().endsWith(json_extension);
  }
}
