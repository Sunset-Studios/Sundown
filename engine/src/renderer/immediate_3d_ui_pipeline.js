import { Buffer } from "./buffer.js";
import { Mesh } from "./mesh.js";
import { MeshData } from "./mesh_data.js";
import { ResourceCache } from "./resource_cache.js";
import { BindGroup } from "./bind_group.js";
import { PipelineState } from "./pipeline_state.js";
import { Shader } from "./shader.js";
import {
  BindGroupType,
  CacheTypes,
  MaterialFamilyType,
  MaterialPassType,
} from "./renderer_types.js";
import { Texture } from "./texture.js";
import { get_ui_3d_commands, UI3DCommandType } from "../ui/3d/immediate.js";
import { Name } from "../utility/names.js";

const invalid_u32 = 0xffffffff;
const instance_stride = 28;
const element_stride = 8;
const string_stride = 8;
const glyph_stride = 4;
const minimum_instance_capacity = 16;
const ui_shader_path = "ui_standard_material.wgsl";
const text_shader_path = "text_material.wgsl";

function ensure_typed_capacity(current, required, type, minimum = minimum_instance_capacity) {
  if (current && current.length >= required) {
    return current;
  }

  const next = new type(Math.max(required, minimum, (current?.length ?? 0) * 2));
  if (current) {
    next.set(current);
  }
  return next;
}

function push_vec4(target, offset, value, fallback = [0, 0, 0, 0]) {
  target[offset + 0] = value?.[0] ?? fallback[0];
  target[offset + 1] = value?.[1] ?? fallback[1];
  target[offset + 2] = value?.[2] ?? fallback[2];
  target[offset + 3] = value?.[3] ?? fallback[3];
}

function add_scaled_axis(out, axis, scale) {
  out[0] += (axis?.[0] ?? 0) * scale;
  out[1] += (axis?.[1] ?? 0) * scale;
  out[2] += (axis?.[2] ?? 0) * scale;
}

function scaled_axis(axis, scale) {
  return [
    (axis?.[0] ?? 0) * scale,
    (axis?.[1] ?? 0) * scale,
    (axis?.[2] ?? 0) * scale,
  ];
}

function offset_origin(origin, x_axis, x_scale, y_axis, y_scale) {
  const next = [
    origin?.[0] ?? 0,
    origin?.[1] ?? 0,
    origin?.[2] ?? 0,
  ];
  add_scaled_axis(next, x_axis, x_scale);
  add_scaled_axis(next, y_axis, y_scale);
  return next;
}

function normalized_rounding(radius, width, height) {
  const size = Math.max(1, Math.min(width || 0, height || 0));
  return Math.min(0.5, Math.max(0, radius || 0) / size);
}

function has_visible_color(color) {
  return (color?.[3] ?? 0) > 0;
}

function draw_quad_range(render_pass, instance_count, first_instance) {
  if (instance_count <= 0) {
    return;
  }

  const index_buffer = MeshData.index_buffer;
  const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;
  const mesh = Mesh.quad();

  render_pass.pass.setIndexBuffer(
    index_buffer.buffer,
    index_buffer.config.element_type,
    mesh.index_buffer_offset * index_buffer_multiplier,
    mesh.index_count * index_buffer_multiplier
  );
  render_pass.pass.drawIndexed(
    mesh.index_count,
    instance_count,
    0,
    mesh.vertex_buffer_offset,
    first_instance
  );
}

function color_targets_from_pass(render_pass) {
  return render_pass.frame_attachments
    .filter((attachment) => !attachment.config.type.includes("depth"))
    .map((attachment) => {
      const target = {
        format: attachment.config.format,
      };
      if (attachment.config.blend) {
        target.blend = attachment.config.blend;
      }
      return target;
    });
}

function depth_target_from_pass(render_pass, pass_type) {
  const depth_attachment = render_pass.frame_attachments.find((attachment) =>
    attachment.config.type.includes("depth")
  );
  if (!depth_attachment) {
    return null;
  }

  return {
    format: depth_attachment.config.format,
    depthWriteEnabled: pass_type !== MaterialPassType.Resolve,
    depthCompare: pass_type === MaterialPassType.Depth ? "less" : "less-equal",
  };
}

export class Immediate3DUIPipeline {
  instance_buffer = null;
  element_buffer = null;
  text_buffer = null;
  string_buffer = null;
  glyph_buffer = null;

  instance_data = null;
  element_data = null;
  text_data = null;
  string_data = null;
  glyph_data = null;

  instance_count = 0;
  batches = [];
  visibility_shader_buckets = [];
  material_bind_groups = new Map();
  pipeline_states = new Map();
  shaders = new Map();

  sort_and_batch() {
    const commands = get_ui_3d_commands().filter((command) => command && !command.deferred);
    if (commands.length === 0) {
      this.instance_count = 0;
      this.batches.length = 0;
      this.visibility_shader_buckets.length = 0;
      return;
    }

    this._prepare_batches(commands);
    this._upload_buffers();
    this._prepare_visibility_buckets();
  }

  get_visibility_shader_buckets() {
    return this.visibility_shader_buckets;
  }

  get_total_draw_count() {
    return this.instance_count;
  }

  get_total_meshlet_count() {
    return this.instance_count;
  }

  get_pass_inputs(render_graph, pass_type) {
    return [];
  }

  submit_visibility_bucket_indirect_draw(render_pass, bucket, _indirect_buffer, pass_type) {
    if (!this._bind_visibility_bucket(render_pass, bucket, pass_type)) {
      return;
    }

    for (const batch_index of bucket.batch_indices) {
      const batch = this.batches[batch_index];
      draw_quad_range(render_pass, batch.count, batch.first);
    }
  }

  submit_visibility_bucket_resolve(render_pass, bucket) {
    if (!this._bind_visibility_bucket(render_pass, bucket, MaterialPassType.Resolve)) {
      return;
    }

    const index_buffer = MeshData.index_buffer;
    const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;
    const mesh = Mesh.quad();
    render_pass.pass.setIndexBuffer(
      index_buffer.buffer,
      index_buffer.config.element_type,
      mesh.index_buffer_offset * index_buffer_multiplier,
      mesh.index_count * index_buffer_multiplier
    );
    render_pass.pass.drawIndexed(mesh.index_count, 1, 0, mesh.vertex_buffer_offset);
  }

  _prepare_batches(commands) {
    commands.sort((a, b) => {
      const type_delta = this._type_sort_key(a) - this._type_sort_key(b);
      if (type_delta !== 0) {
        return type_delta;
      }
      const font_delta = String(a.font_texture ?? "").localeCompare(String(b.font_texture ?? ""));
      if (font_delta !== 0) {
        return font_delta;
      }
      const z_delta = (a.z_order ?? 0) - (b.z_order ?? 0);
      return z_delta !== 0 ? z_delta : (a.order ?? 0) - (b.order ?? 0);
    });

    const max_instance_count = commands.length * 5;
    this.instance_data = ensure_typed_capacity(
      this.instance_data,
      max_instance_count * instance_stride,
      Float32Array,
      minimum_instance_capacity * instance_stride
    );
    this.element_data = ensure_typed_capacity(
      this.element_data,
      max_instance_count * element_stride,
      Float32Array,
      minimum_instance_capacity * element_stride
    );
    this.text_data = ensure_typed_capacity(this.text_data, max_instance_count, Uint32Array);
    this.string_data = ensure_typed_capacity(
      this.string_data,
      max_instance_count * string_stride,
      Float32Array,
      minimum_instance_capacity * string_stride
    );
    this.glyph_data = ensure_typed_capacity(
      this.glyph_data,
      max_instance_count * glyph_stride,
      Int32Array,
      minimum_instance_capacity * glyph_stride
    );

    this.instance_data.fill(0, 0, max_instance_count * instance_stride);
    this.element_data.fill(0, 0, max_instance_count * element_stride);
    this.text_data.fill(0, 0, max_instance_count);
    this.string_data.fill(0, 0, max_instance_count * string_stride);
    this.glyph_data.fill(0, 0, max_instance_count * glyph_stride);

    this.instance_count = 0;
    this.batches.length = 0;

    let current_batch = null;
    for (const command of commands) {
      if (command.type === UI3DCommandType.Quad) {
        current_batch = this._ensure_batch(current_batch, command.type);
        const first = this.instance_count;
        this._write_quad_command(command);
        current_batch.count += this.instance_count - first;
      } else if (command.type === UI3DCommandType.Text && command.font_texture) {
        current_batch = this._ensure_batch(current_batch, command.type, command.font_texture);
        this._write_text(command, this.instance_count++);
        current_batch.count++;
      }
    }
  }

  _prepare_visibility_buckets() {
    this.visibility_shader_buckets.length = 0;
    const buckets_by_key = new Map();

    for (let i = 0; i < this.batches.length; i++) {
      const batch = this.batches[i];
      if (batch.count <= 0) {
        continue;
      }

      const key_name = batch.type === UI3DCommandType.Text
        ? `Immediate3DUI|Text|${batch.font_texture}`
        : "Immediate3DUI|Quad";
      const key = Name.from(key_name);
      let bucket = buckets_by_key.get(key);
      if (!bucket) {
        bucket = {
          key,
          shader: this._get_shader(batch, MaterialPassType.Raster),
          depth_shader: this._get_shader(batch, MaterialPassType.Depth),
          resolve_shader: this._get_shader(batch, MaterialPassType.Resolve),
          template_name: key_name,
          representative_material_id: 0,
          family: MaterialFamilyType.Transparent,
          queue: this,
          batch_type: batch.type,
          font_texture: batch.font_texture,
          batch_indices: [],
        };
        buckets_by_key.set(key, bucket);
        this.visibility_shader_buckets.push(bucket);
      }

      bucket.batch_indices.push(i);
    }
  }

  _type_sort_key(command) {
    return command.type === UI3DCommandType.Text ? 1 : 0;
  }

  _ensure_batch(current_batch, type, font_texture = null) {
    const should_start_batch =
      !current_batch ||
      current_batch.type !== type ||
      current_batch.font_texture !== font_texture;

    if (!should_start_batch) {
      return current_batch;
    }

    const batch = {
      type,
      font_texture,
      first: this.instance_count,
      count: 0,
    };
    this.batches.push(batch);
    return batch;
  }

  _write_quad_command(command) {
    const border_width = Math.min(
      Math.max(0, command.border_width ?? 0),
      Math.max(0, command.width ?? 0) * 0.5,
      Math.max(0, command.height ?? 0) * 0.5
    );
    const has_border = border_width > 0 && has_visible_color(command.border_color);

    if (has_border) {
      this._write_border_quads(command, border_width);
    }

    if (has_visible_color(command.fill_color)) {
      if (has_border) {
        const x_scale = border_width / Math.max(1, command.width);
        const y_scale = border_width / Math.max(1, command.height);
        const inner_width = Math.max(0, command.width - border_width * 2);
        const inner_height = Math.max(0, command.height - border_width * 2);
        if (inner_width > 0 && inner_height > 0) {
          this._write_quad_instance({
            origin: offset_origin(command.origin, command.x_axis, x_scale, command.y_axis, y_scale),
            x_axis: scaled_axis(command.x_axis, Math.max(0, 1 - x_scale * 2)),
            y_axis: scaled_axis(command.y_axis, Math.max(0, 1 - y_scale * 2)),
            color: command.fill_color,
            width: inner_width,
            height: inner_height,
            corner_radius: Math.max(0, (command.corner_radius ?? 0) - border_width),
            emissive: command.emissive,
          }, this.instance_count++);
        }
      } else {
        this._write_quad_instance({
          origin: command.origin,
          x_axis: command.x_axis,
          y_axis: command.y_axis,
          color: command.fill_color,
          width: command.width,
          height: command.height,
          corner_radius: command.corner_radius,
          emissive: command.emissive,
        }, this.instance_count++);
      }
    }
  }

  _write_border_quads(command, border_width) {
    const width = Math.max(1, command.width);
    const height = Math.max(1, command.height);
    const x_scale = border_width / width;
    const y_scale = border_width / height;
    const middle_height_scale = Math.max(0, 1 - y_scale * 2);
    const right_x = Math.max(0, 1 - x_scale);
    const bottom_y = Math.max(0, 1 - y_scale);

    this._write_quad_instance({
      origin: command.origin,
      x_axis: command.x_axis,
      y_axis: scaled_axis(command.y_axis, y_scale),
      color: command.border_color,
      width,
      height: border_width,
      corner_radius: command.corner_radius,
      emissive: command.emissive,
    }, this.instance_count++);

    this._write_quad_instance({
      origin: offset_origin(command.origin, command.x_axis, 0, command.y_axis, bottom_y),
      x_axis: command.x_axis,
      y_axis: scaled_axis(command.y_axis, y_scale),
      color: command.border_color,
      width,
      height: border_width,
      corner_radius: command.corner_radius,
      emissive: command.emissive,
    }, this.instance_count++);

    if (middle_height_scale > 0) {
      this._write_quad_instance({
        origin: offset_origin(command.origin, command.x_axis, 0, command.y_axis, y_scale),
        x_axis: scaled_axis(command.x_axis, x_scale),
        y_axis: scaled_axis(command.y_axis, middle_height_scale),
        color: command.border_color,
        width: border_width,
        height: Math.max(0, height - border_width * 2),
        corner_radius: 0,
        emissive: command.emissive,
      }, this.instance_count++);

      this._write_quad_instance({
        origin: offset_origin(command.origin, command.x_axis, right_x, command.y_axis, y_scale),
        x_axis: scaled_axis(command.x_axis, x_scale),
        y_axis: scaled_axis(command.y_axis, middle_height_scale),
        color: command.border_color,
        width: border_width,
        height: Math.max(0, height - border_width * 2),
        corner_radius: 0,
        emissive: command.emissive,
      }, this.instance_count++);
    }
  }

  _write_quad_instance(command, instance_index) {
    const instance_offset = instance_index * instance_stride;
    push_vec4(this.instance_data, instance_offset + 0, command.origin, [0, 0, 0, 1]);
    push_vec4(this.instance_data, instance_offset + 4, command.x_axis, [1, 0, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 8, command.y_axis, [0, -1, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 12, [0, 0, 1, 0]);
    push_vec4(this.instance_data, instance_offset + 16, [0, 0, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 20, [0, 0, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 24, [
      command.width,
      command.height,
      command.corner_radius,
      0,
    ]);

    const element_offset = instance_index * element_stride;
    push_vec4(this.element_data, element_offset + 0, command.color, [1, 1, 1, 1]);
    this.element_data[element_offset + 4] = command.emissive ?? 0;
    this.element_data[element_offset + 5] = normalized_rounding(
      command.corner_radius,
      command.width,
      command.height
    );
  }

  _write_text(command, instance_index) {
    const instance_offset = instance_index * instance_stride;
    push_vec4(this.instance_data, instance_offset + 0, command.origin, [0, 0, 0, 1]);
    push_vec4(this.instance_data, instance_offset + 4, command.x_axis, [1, 0, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 8, command.y_axis, [0, -1, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 12, [0, 0, 1, 1]);
    push_vec4(this.instance_data, instance_offset + 16, [0, 0, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 20, [0, 0, 0, 0]);
    push_vec4(this.instance_data, instance_offset + 24, [0, 0, 0, 0]);

    this.text_data[instance_index] = instance_index;

    const string_offset = instance_index * string_stride;
    push_vec4(this.string_data, string_offset + 0, command.color, [1, 1, 1, 1]);
    this.string_data[string_offset + 4] = command.page_texture_size?.[0] ?? 1;
    this.string_data[string_offset + 5] = command.page_texture_size?.[1] ?? 1;
    this.string_data[string_offset + 6] = command.emissive ?? 1;

    const glyph_offset = instance_index * glyph_stride;
    this.glyph_data[glyph_offset + 0] = command.glyph_width ?? 0;
    this.glyph_data[glyph_offset + 1] = command.glyph_height ?? 0;
    this.glyph_data[glyph_offset + 2] = command.glyph_x ?? 0;
    this.glyph_data[glyph_offset + 3] = command.glyph_y ?? 0;
  }

  _upload_buffers() {
    this.instance_buffer = this._create_or_update_buffer(
      "immediate_3d_ui_instances",
      this.instance_data,
      Math.max(this.instance_count * instance_stride, instance_stride),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.element_buffer = this._create_or_update_buffer(
      "immediate_3d_ui_element_data",
      this.element_data,
      Math.max(this.instance_count * element_stride, element_stride),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.text_buffer = this._create_or_update_buffer(
      "immediate_3d_ui_text_indices",
      this.text_data,
      Math.max(this.instance_count, 1),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.string_buffer = this._create_or_update_buffer(
      "immediate_3d_ui_string_data",
      this.string_data,
      Math.max(this.instance_count * string_stride, string_stride),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.glyph_buffer = this._create_or_update_buffer(
      "immediate_3d_ui_glyph_data",
      this.glyph_data,
      Math.max(this.instance_count * glyph_stride, glyph_stride),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
  }

  _create_or_update_buffer(name, data, element_count, usage) {
    const buffer = Buffer.create({
      name,
      raw_data: data,
      usage,
    });
    buffer.write_raw(data, 0, element_count, 0);
    return buffer;
  }

  _bind_visibility_bucket(render_pass, bucket, pass_type) {
    const bind_groups = render_pass.frame_bind_groups;
    const material_bind_group = this._get_material_bind_group_for_bucket(bucket);
    const pipeline = this._get_pipeline_for_bucket(render_pass, bucket, material_bind_group, pass_type);
    if (!pipeline?.is_ready()) {
      return false;
    }

    if (bind_groups[BindGroupType.Global]) {
      bind_groups[BindGroupType.Global].bind(render_pass);
    }
    if (bind_groups[BindGroupType.Pass]) {
      bind_groups[BindGroupType.Pass].bind(render_pass);
    }
    material_bind_group.bind(render_pass);
    render_pass.set_pipeline(pipeline);
    return true;
  }

  _get_material_bind_group_for_bucket(bucket) {
    const cache_key = bucket.batch_type === UI3DCommandType.Text
      ? `text|${bucket.font_texture}`
      : "quad";
    const cached = this.material_bind_groups.get(cache_key);
    if (cached) {
      return cached;
    }

    const bind_group = bucket.batch_type === UI3DCommandType.Text
      ? this._create_text_material_bind_group(bucket.font_texture)
      : this._create_ui_material_bind_group();
    this.material_bind_groups.set(cache_key, bind_group);
    return bind_group;
  }

  _create_ui_material_bind_group() {
    return BindGroup.create_with_layout(
      "immediate_3d_ui_standard_material_bindgroup",
      [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
      ],
      BindGroupType.Material,
      [
        {
          binding: 0,
          resource: {
            buffer: this.element_buffer.buffer,
            offset: 0,
            size: this.element_buffer.config.size,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.instance_buffer.buffer,
            offset: 0,
            size: this.instance_buffer.config.size,
          },
        },
      ],
      true
    );
  }

  _create_text_material_bind_group(font_texture_name) {
    const texture =
      ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(font_texture_name)) ||
      Texture.default();

    return BindGroup.create_with_layout(
      `immediate_3d_ui_text_material_${font_texture_name}_bindgroup`,
      [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: {
            viewDimension: texture.config.dimension,
            sampleType: Texture.filter_type_from_format(texture.config.format),
          },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "read-only-storage",
          },
        },
      ],
      BindGroupType.Material,
      [
        {
          binding: 0,
          resource: {
            buffer: this.text_buffer.buffer,
            offset: 0,
            size: this.text_buffer.config.size,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.string_buffer.buffer,
            offset: 0,
            size: this.string_buffer.config.size,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.glyph_buffer.buffer,
            offset: 0,
            size: this.glyph_buffer.config.size,
          },
        },
        {
          binding: 3,
          resource: texture.view,
        },
        {
          binding: 4,
          resource: {
            buffer: this.instance_buffer.buffer,
            offset: 0,
            size: this.instance_buffer.config.size,
          },
        },
      ],
      true
    );
  }

  _get_pipeline_for_bucket(render_pass, bucket, material_bind_group, pass_type) {
    const cache_key = [
      bucket.key,
      pass_type,
      render_pass.frame_attachments.map((attachment) => attachment.config.name).join("|"),
    ].join("|");
    const cached = this.pipeline_states.get(cache_key);
    if (cached) {
      return cached;
    }

    const shader = pass_type === MaterialPassType.Depth
      ? bucket.depth_shader
      : pass_type === MaterialPassType.Resolve
        ? bucket.resolve_shader
        : bucket.shader;
    const depth_target = depth_target_from_pass(render_pass, pass_type);
    const descriptor = {
      bind_layouts: [
        render_pass.frame_bind_groups[BindGroupType.Global]?.layout,
        render_pass.frame_bind_groups[BindGroupType.Pass]?.layout,
        material_bind_group.layout,
      ].filter((layout) => layout !== null && layout !== undefined),
      vertex: {
        module: shader.module,
        entryPoint: "vs",
        buffers: [],
      },
      fragment: {
        module: shader.module,
        entryPoint: "fs",
        targets: color_targets_from_pass(render_pass),
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
      defer_creation: false,
    };

    if (depth_target) {
      descriptor.depthStencil = depth_target;
    }

    const pipeline = PipelineState.create_render(`Immediate3DUI_${cache_key}`, descriptor);
    this.pipeline_states.set(cache_key, pipeline);
    return pipeline;
  }

  _get_shader(batch, pass_type) {
    const shader_path = batch.type === UI3DCommandType.Text ? text_shader_path : ui_shader_path;
    const define_name = pass_type === MaterialPassType.Depth
      ? "MESHLET_DEPTH_PASS"
      : pass_type === MaterialPassType.Resolve
        ? "MESHLET_RESOLVE_PASS"
        : "MESHLET_RASTER_PASS";
    const cache_key = `${shader_path}|${define_name}`;
    const cached = this.shaders.get(cache_key);
    if (cached) {
      return cached;
    }

    const shader_id = batch.type === UI3DCommandType.Text
      ? this._create_text_shader(pass_type)
      : this._create_ui_shader(pass_type);
    const shader = ResourceCache.get().fetch(CacheTypes.SHADER, shader_id);
    this.shaders.set(cache_key, shader);
    return shader;
  }

  _create_ui_shader(pass_type) {
    if (pass_type === MaterialPassType.Depth) {
      return Shader.create(ui_shader_path, {
        MESHLET_DEPTH_PASS: true,
        TRANSPARENT: true,
      });
    }
    if (pass_type === MaterialPassType.Resolve) {
      return Shader.create(ui_shader_path, {
        MESHLET_RESOLVE_PASS: true,
        TRANSPARENT: true,
      });
    }
    return Shader.create(ui_shader_path, {
      MESHLET_RASTER_PASS: true,
      TRANSPARENT: true,
    });
  }

  _create_text_shader(pass_type) {
    if (pass_type === MaterialPassType.Depth) {
      return Shader.create(text_shader_path, {
        MESHLET_DEPTH_PASS: true,
        TRANSPARENT: true,
      });
    }
    if (pass_type === MaterialPassType.Resolve) {
      return Shader.create(text_shader_path, {
        MESHLET_RESOLVE_PASS: true,
        TRANSPARENT: true,
      });
    }
    return Shader.create(text_shader_path, {
      MESHLET_RASTER_PASS: true,
      TRANSPARENT: true,
    });
  }
}
