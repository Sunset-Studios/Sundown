import { Texture } from "../../renderer/texture.js";
import { Name } from "../../utility/names.js";
import { read_file } from "../../utility/file_system.js";

const chars_key = 'chars';
const pages_key = 'pages';
const id_key = 'id';
const width_key = 'width';
const height_key = 'height';
const x_offset_key = 'xoffset';
const y_offset_key = 'yoffset';
const x_advance_key = 'xadvance';
const page_key = 'page';
const x_key = 'x';
const y_key = 'y';

const page_format = 'rgba8unorm';
const page_dimension = '2d';
const path_sep = '/';
const extension_sep = '.';

export class Font {
  code_point = null;
  width = null;
  height = null;
  x_offset = null;
  y_offset = null;
  x_advance = null;
  page = null;
  x = null;
  y = null;
  page_textures = null;

  constructor(num_chars) {
    this.code_point = new Uint32Array(num_chars);
    this.width = new Uint16Array(num_chars);
    this.height = new Uint16Array(num_chars);
    this.x_offset = new Int16Array(num_chars);
    this.y_offset = new Int16Array(num_chars);
    this.x_advance = new Uint16Array(num_chars);
    this.page = new Uint16Array(num_chars);
    this.x = new Int16Array(num_chars);
    this.y = new Int16Array(num_chars);
  }

  static create(context, font_data_file) {
    const font_data = JSON.parse(read_file(font_data_file));
    if (!font_data) return null;

    const font = new Font(font_data[chars_key].length);

    for (let i = 0; i < font_data[chars_key].length; i++) {
      const char = font_data[chars_key][i];
      font.code_point[i] = char[id_key];
      font.width[i] = char[width_key];
      font.height[i] = char[height_key];
      font.x_offset[i] = char[x_offset_key];
      font.y_offset[i] = char[y_offset_key];
      font.x_advance[i] = char[x_advance_key];
      font.page[i] = char[page_key];
      font.x[i] = char[x_key];
      font.y[i] = char[y_key];
    }

    font.page_textures = new Uint32Array(font_data[pages_key].length);
    for (let i = 0; i < font_data[pages_key].length; i++) {
      const page = font_data[pages_key][i];
      const page_location = font_data_file.substring(0, font_data_file.lastIndexOf(path_sep) + 1) + page;
      const page_name = page.substring(0, page.lastIndexOf(extension_sep));
      Texture.load(context, [page_location], {
        name: page_name,
        format: page_format,
        dimension: page_dimension,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      font.page_textures[i] = Name.from(page_name);
    }

    return font;
  }
}
