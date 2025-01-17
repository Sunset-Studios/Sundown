import { Texture } from "../../renderer/texture.js";
import { Buffer } from "../../renderer/buffer.js";
import { Material, MaterialTemplate } from "../../renderer/material.js";
import { ResourceCache } from "../../renderer/resource_cache.js";
import { Name } from "../../utility/names.js";
import { read_file } from "../../utility/file_system.js";
import { no_cull_rasterizer_config } from "../../utility/config_permutations.js";
import { CacheTypes, MaterialFamilyType } from "../../renderer/renderer_types.js";
import { SquareAdjacencyMatrix } from "../../memory/container.js";

const chars_key = "chars";
const common_key = "common";
const pages_key = "pages";
const id_key = "id";
const width_key = "width";
const height_key = "height";
const x_offset_key = "xoffset";
const y_offset_key = "yoffset";
const x_advance_key = "xadvance";
const kernings_key = "kernings";
const page_key = "page";
const x_key = "x";
const y_key = "y";

const first_key = "first";
const second_key = "second";
const amount_key = "amount";

const text_data_key = "text";
const string_data_key = "string_data";
const font_glyph_data_key = "font_glyph_data";
const font_page_texture_key = "font_page_texture";

const material_key = "material";
const default_text_material_template_key = "DefaultTextMaterial";
const default_text_shader_key = "text_material.wgsl";

const page_format = "rgba8unorm";
const page_dimension = "2d";
const path_sep = "/";
const extension_sep = ".";
const underscore = "_";

export class Font {
  code_point_index_map = null;
  code_point = null;
  width = null;
  height = null;
  x_offset = null;
  y_offset = null;
  x_advance = null;
  page = null;
  x = null;
  y = null;
  texture_width = 0;
  texture_height = 0;

  sizes_and_offsets_buffer = null;
  positions_and_advance_and_page_buffer = null;

  page_textures = null;

  material = null;

  constructor(num_chars) {
    this.code_point_index_map = new Map();
    this.code_point = new Uint32Array(num_chars);
    this.width = new Uint16Array(num_chars);
    this.height = new Uint16Array(num_chars);
    this.x_offset = new Int16Array(num_chars);
    this.y_offset = new Int16Array(num_chars);
    this.x_advance = new Uint16Array(num_chars);
    this.page = new Uint16Array(num_chars);
    this.x = new Int16Array(num_chars);
    this.y = new Int16Array(num_chars);
    this.kerning_matrix = null;
    this.line_height = 0;
  }

  static create(font_data_file) {
    const template = MaterialTemplate.get_template(default_text_material_template_key);
    if (!template) {
      MaterialTemplate.create(
        default_text_material_template_key,
        default_text_shader_key,
        MaterialFamilyType.Opaque,
        no_cull_rasterizer_config
      );
    }

    const font_data = JSON.parse(read_file(font_data_file));
    if (!font_data) return null;

    const last_sep_index = font_data_file.lastIndexOf(path_sep);
    const extension_index = font_data_file.lastIndexOf(extension_sep);
    const font_name = font_data_file.substring(last_sep_index + 1, extension_index);
    const font_name_suffix = font_name + underscore;

    const font = new Font(font_data[chars_key].length);

    font.texture_width = font_data[common_key].scaleW;
    font.texture_height = font_data[common_key].scaleH;
    font.line_height = font_data[common_key].lineHeight;

    for (let i = 0; i < font_data[chars_key].length; i++) {
      const char = font_data[chars_key][i];
      font.code_point_index_map.set(char[id_key], i);
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

    font.kerning_matrix = new SquareAdjacencyMatrix(font.code_point);
    for (let i = 0; i < font_data[kernings_key].length; ++i) {
      const kerning = font_data[kernings_key][i];
      font.kerning_matrix.set_adjacent_value(
        kerning[first_key],
        kerning[second_key],
        kerning[amount_key]
      );
    }

    const font_glyph_data = new Int32Array(font_data[chars_key].length * 4);
    for (let i = 0; i < font_data[chars_key].length; i++) {
      font_glyph_data[i * 4] = font.width[i];
      font_glyph_data[i * 4 + 1] = font.height[i];
      font_glyph_data[i * 4 + 2] = font.x[i];
      font_glyph_data[i * 4 + 3] = font.y[i];
    }
    font.font_glyph_data_buffer = Buffer.create({
      name: font_name_suffix + font_glyph_data_key,
      raw_data: font_glyph_data,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    font.page_textures = new Float64Array(font_data[pages_key].length);
    for (let i = 0; i < font_data[pages_key].length; i++) {
      const page = font_data[pages_key][i];
      const page_location =
        font_data_file.substring(0, font_data_file.lastIndexOf(path_sep) + 1) + page;
      const page_name = page.substring(0, page.lastIndexOf(extension_sep));
      Texture.load([page_location], {
        name: page_name,
        format: page_format,
        dimension: page_dimension,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
        force: true,
      });
      font.page_textures[i] = Name.from(page_name);
    }

    font.material = Material.create(
      font_name_suffix + material_key,
      default_text_material_template_key,
      {
        family: MaterialFamilyType.Opaque,
      }
    );

    const material_obj = Material.get(font.material);
    const page_texture_obj = ResourceCache.get().fetch(
      CacheTypes.IMAGE,
      Number(font.page_textures[0])
    );

    material_obj.set_texture_data(font_page_texture_key, page_texture_obj);
    material_obj.set_storage_data(font_glyph_data_key, font.font_glyph_data_buffer);
    material_obj.listen_for_storage_data(text_data_key);
    material_obj.listen_for_storage_data(string_data_key);

    return font;
  }
}
