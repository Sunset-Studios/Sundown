import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { Renderer } from "../renderer.js";
import { BufferSync } from "../buffer.js";
import { ResourceCache } from "../resource_cache.js";
import { RenderPassFlags, DebugDrawType, CacheTypes } from "../renderer_types.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import {
  rgba8unorm_format,
  rgba16float_format,
  r32uint_format,
  depth24plus_format,
  load_op_load,
} from "../../utility/config_permutations.js";
import { Name } from "../../utility/names.js";

// ============ GPU Resource Configs ============

const atlas_config = {
  name: "shadow_atlas",
  format: depth24plus_format,
  dimension: "2d-array",
  width: 0,
  height: 0,
  depth: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  load_op: load_op_load,
};

const page_table_config = {
  name: "shadow_page_table",
  format: r32uint_format,
  dimension: "2d-array",
  width: 0,
  height: 0,
  depth: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
  load_op: load_op_load,
};

const bitmask_buf_config = {
  name: "shadow_bitmask_buf",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const requested_tiles_buf_config = {
  name: "shadow_requested_tiles_buf",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  cpu_readback: true,
};

const settings_buf_config = {
  name: "shadow_settings_buf",
  raw_data: new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]),
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const lru_buf_config = {
  name: "shadow_lru_buf",
  raw_data: null,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const physical_to_virtual_map_buf_config = {
  name: "shadow_physical_to_virtual_map_buf",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const debug_shadow_atlas_config = {
  name: "debug_shadow_atlas",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
};

const debug_page_table_config = {
  name: "debug_page_table",
  format: rgba8unorm_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
};

const debug_tile_overlay_config = {
  name: "debug_tile_overlay",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
};

const tile_request_uniform_configs = [];
// ============ Shader Setup ============

const feedback_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/feedback.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const gather_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/gather.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const page_table_update_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/page_table_update.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const render_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "shadow/as_vsm/tile_render.vert.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const debug_shadow_atlas_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "shadow/as_vsm/debug_shadow_atlas.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const debug_page_table_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "shadow/as_vsm/debug_page_table.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const debug_tile_overlay_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "shadow/as_vsm/debug_tile_overlay.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const MAX_TILE_REQUESTS_PER_VIEW = 64;
const MAX_NUM_TEXTURE_POOLS = 1;

// =============================================================
//  Page-table entry (32-bit) bit-field layout
//  [0 – 6]   : physical page X index   (7 bits)  (0-127)
//  [7 – 13]  : physical page Y index   (7 bits)  (0-127)
//  [14 – 16] : atlas / memory-pool id  (3 bits)  (0-7)
//  [17]      : residency flag          (1 bit)   (1 = resident)
//  [18]      : dirty flag              (1 bit)   (1 = needs update)
//  [19 – 26] : frame age / marker      (8 bits)  (wraps every 256 frames)
//  [27 – 31] : reserved / unused
// =============================================================

// Masks & shifts (snake_case as per project style guide)
const phys_x_shift = 0;
const phys_x_mask  = 0x7f << phys_x_shift; // 7 bits
const phys_y_shift = 7;
const phys_y_mask  = 0x7f << phys_y_shift;
const pool_id_shift = 14;
const pool_id_mask  = 0x7 << pool_id_shift; // 3 bits
const residency_shift = 17;
const residency_mask  = 0x1 << residency_shift;
const dirty_shift = 18;
const dirty_mask  = 0x1 << dirty_shift;
const frame_age_shift = 19;
const frame_age_mask  = 0xff << frame_age_shift; // 8 bits

export function encode_page_table_entry({
  physical_x = 0,
  physical_y = 0,
  pool_id = 0,
  resident = 0,
  dirty = 1,
  frame_age = 0,
} = {}) {
  // Clamp inputs to valid ranges
  physical_x &= 0x7f;
  physical_y &= 0x7f;
  pool_id    &= 0x7;
  frame_age  &= 0xff;

  return (
    (physical_x << phys_x_shift) |
    (physical_y << phys_y_shift) |
    (pool_id    << pool_id_shift) |
    (resident   ? residency_mask : 0) |
    (dirty      ? dirty_mask     : 0) |
    (frame_age  << frame_age_shift)
  ) >>> 0; // ensure unsigned 32-bit
}

export function decode_page_table_entry(entry) {
  return {
    physical_x : (entry & phys_x_mask)  >>> phys_x_shift,
    physical_y : (entry & phys_y_mask)  >>> phys_y_shift,
    pool_id    : (entry & pool_id_mask) >>> pool_id_shift,
    resident   : (entry & residency_mask) !== 0,
    dirty      : (entry & dirty_mask)     !== 0,
    frame_age  : (entry & frame_age_mask) >>> frame_age_shift,
  };
}

/**
 * Adaptive Sparse Virtual Shadow Maps (AS-VSM)
 * Scaffolding: allocates GPU resources and adds stub passes.
 */
export class AdaptiveSparseVirtualShadowMaps {
  static all_instances = [];

  constructor({ atlas_size, tile_size, virtual_dim, max_lods, clip0_extent }) {
    this.tile_size = tile_size;
    this.virtual_dim = virtual_dim;
    this.atlas_size = atlas_size;
    this.max_lods = max_lods;
    this.clip0_extent = clip0_extent;
    this.virtual_tiles_per_row = Math.ceil(this.virtual_dim / this.tile_size);
    this.total_virtual_tiles =
      this.virtual_tiles_per_row * this.virtual_tiles_per_row * this.max_lods;
    this.physical_tiles_per_row = Math.ceil(this.atlas_size / this.tile_size);
    this.total_physical_tiles =
      this.physical_tiles_per_row * this.physical_tiles_per_row * this.max_lods;
    this.cached_light_count = null;

    const num_elements = 1 + MAX_TILE_REQUESTS_PER_VIEW * 3;
    this.cpu_requested_tiles = new Uint32Array(num_elements);

    AdaptiveSparseVirtualShadowMaps.all_instances.push(this);
  }

  add_passes(
    render_graph,
    {
      depth_texture,
      position_texture,
      lights_buffer,
      dense_shadow_casting_lights_buffer,
      light_count_buffer,
      transforms_buffer,
      object_instances,
      view_visibility_buffers,
      force_recreate = false,
      debug_view = null,
    }
  ) {
    const max_light_count = LightFragment.total_shadow_casting_lights;

    // Insert caching check for the number of lights
    if (this.cached_light_count === null || this.cached_light_count !== max_light_count) {
      this.cached_light_count = max_light_count;
      force_recreate |= true;
    }

    const adjusted_light_count = Math.max(this.cached_light_count, 1);

    settings_buf_config.force = force_recreate;
    this.settings_buf = render_graph.create_buffer(settings_buf_config);

    // Create Bitmask buffer
    const bitmask_u32_count = Math.ceil(this.total_virtual_tiles) >> 5; // total_tiles / 32
    bitmask_buf_config.size = bitmask_u32_count * 4;
    bitmask_buf_config.force = force_recreate;
    this.bitmask_buf = render_graph.create_buffer(bitmask_buf_config);
    this.bitmask_u32_count = bitmask_u32_count;

    // Create Requested Tiles buffer
    this.max_tile_requests = MAX_TILE_REQUESTS_PER_VIEW;
    requested_tiles_buf_config.size = adjusted_light_count * this.max_tile_requests * 3 * 4 + 4; // 64 tile requests per view (3 u32 per request);
    requested_tiles_buf_config.force = force_recreate;
    this.requested_tiles_buf = render_graph.create_buffer(requested_tiles_buf_config);

    // Create Physical Shadow Atlas texture array
    atlas_config.width = this.atlas_size;
    atlas_config.height = this.atlas_size;
    atlas_config.depth = MAX_NUM_TEXTURE_POOLS;
    atlas_config.force = force_recreate;
    atlas_config.b_one_view_per_layer = true;
    this.shadow_atlas = render_graph.create_image(atlas_config);

    // Create Page Table storage texture
    page_table_config.width = this.virtual_tiles_per_row;
    page_table_config.height = this.virtual_tiles_per_row;
    page_table_config.depth = adjusted_light_count * this.max_lods;
    page_table_config.force = force_recreate;
    this.page_table = render_graph.create_image(page_table_config);

    // Create Physical to Virtual map buffer
    const physical_tiles_per_view =
      this.physical_tiles_per_row * this.physical_tiles_per_row;
    // Store per-view physical tile count
    this.total_physical_tiles = physical_tiles_per_view;
    physical_to_virtual_map_buf_config.size = adjusted_light_count * physical_tiles_per_view * 4;
    physical_to_virtual_map_buf_config.force = force_recreate;
    this.physical_to_virtual_map_buf = render_graph.create_buffer(
      physical_to_virtual_map_buf_config
    );

    // Create LRU ring buffer
    if (force_recreate) {
      const lru_per_light = physical_tiles_per_view;
      const total_lru_entries = adjusted_light_count * (lru_per_light + 1);
      const lru_raw = new Uint32Array(total_lru_entries);
      for (let light_idx = 0; light_idx < adjusted_light_count; light_idx++) {
        const offset = light_idx * (lru_per_light + 1);
        lru_raw[offset] = 0;
        for (let i = 1; i <= lru_per_light; i++) {
          lru_raw[offset + i] = i - 1;
        }
      }
      lru_buf_config.raw_data = lru_raw;
    }

    lru_buf_config.force = force_recreate;
    this.lru_buf = render_graph.create_buffer(lru_buf_config);
    // Can discard lru_raw now that buffer is created
    lru_buf_config.raw_data = null;

    // Store references for debug passes
    this.position_texture = position_texture;
    this.lights_buffer = lights_buffer;

    // Stage 0: Clear buffers
    render_graph.add_pass(
      "as_vsm_init",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        // Clear settings buffer
        const settings = graph.get_physical_buffer(this.settings_buf);
        settings.write_raw(
          new Float32Array([
            this.tile_size, // tile_size
            this.virtual_dim, // virtual_dim
            this.virtual_tiles_per_row, // virtual_tiles_per_row
            this.atlas_size, // atlas_size
            this.physical_tiles_per_row, // physical_tiles_per_row
            this.max_lods, // max_lod
            this.max_tile_requests, // max_tile_requests
            this.clip0_extent, // clip0_extent
          ])
        );

        // Clear bitmask buffer
        const bitmask = graph.get_physical_buffer(this.bitmask_buf);
        bitmask.write_raw(new Uint32Array(this.bitmask_u32_count));

        // Clear requested tiles buffer
        const requested_tiles = graph.get_physical_buffer(this.requested_tiles_buf);
        requested_tiles.write_raw(new Uint32Array([0]));
      }
    );

    // Stage A: Screen-space Feedback
    render_graph.add_pass(
      "as_vsm_feedback",
      RenderPassFlags.Compute,
      {
        inputs: [
          depth_texture,
          this.shadow_atlas,
          this.page_table,
          this.settings_buf,
          this.bitmask_buf,
          lights_buffer,
          dense_shadow_casting_lights_buffer,
          light_count_buffer,
        ],
        outputs: [this.bitmask_buf],
        shader_setup: feedback_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const depth_img = graph.get_physical_image(depth_texture);
        const w = depth_img.config.width;
        const h = depth_img.config.height;
        const light_groups = Math.ceil(adjusted_light_count / 4);
        pass.dispatch(Math.ceil(w / 8), Math.ceil(h / 8), light_groups);
      }
    );

    // Stage B: New Tile Gather
    render_graph.add_pass(
      "as_vsm_gather",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.bitmask_buf,
          this.shadow_atlas,
          this.page_table,
          this.requested_tiles_buf,
          lights_buffer,
          dense_shadow_casting_lights_buffer,
          light_count_buffer,
          this.settings_buf,
        ],
        outputs: [this.requested_tiles_buf],
        shader_setup: gather_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const bitmask_groups = Math.ceil(this.bitmask_u32_count / 8);
        const light_groups = Math.ceil(adjusted_light_count / 4);
        pass.dispatch(1, bitmask_groups, light_groups);
      }
    );

    // Stage C: Update Page Table
    render_graph.add_pass(
      "as_vsm_update_page_table",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.requested_tiles_buf,
          this.shadow_atlas,
          this.lru_buf,
          this.page_table,
          dense_shadow_casting_lights_buffer,
          this.settings_buf,
          this.physical_to_virtual_map_buf,
        ],
        outputs: [this.page_table, this.physical_to_virtual_map_buf],
        shader_setup: page_table_update_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const requested_tiles_groups = Math.ceil(MAX_TILE_REQUESTS_PER_VIEW / 64);
        pass.dispatch(requested_tiles_groups, 1, 1);
      }
    );

    // Stage D: Process tile requests using data read back from the GPU in the *previous* frame.
    const active_tile_count = Math.min(this.cpu_requested_tiles[0], MAX_TILE_REQUESTS_PER_VIEW);

    // Allocate / reuse uniform buffers for each active tile request.
    const tile_request_uniforms = this._setup_tile_request_uniforms(
      render_graph,
      active_tile_count
    );

    for (let i = 0; i < active_tile_count; i++) {
      const request_index = 1 + i * 3;
      const view_index = this.cpu_requested_tiles[request_index + 2];
      const visible_object_instances = view_visibility_buffers[view_index];

      render_graph.add_pass(
        `as_vsm_render_tile_${i}`,
        RenderPassFlags.Graphics,
        {
          inputs: [
            transforms_buffer,
            object_instances,
            visible_object_instances,
            this.requested_tiles_buf,
            dense_shadow_casting_lights_buffer,
            this.settings_buf,
            this.page_table,
            tile_request_uniforms[i],
          ],
          outputs: [this.shadow_atlas],
          shader_setup: render_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.submit_indexed_indirect_draws(
            pass,
            view_index,
            true /* skip_material_bind */
          );
        }
      );
    }

    // Debug AS-VSM views
    if (debug_view !== DebugDrawType.None) {
      this.add_debug_passes(render_graph, force_recreate, debug_view);
    }

    // Schedule an asynchronous readback so the next frame has up-to-date request data.
    BufferSync.request_readback(this);
  }

  async readback_buffers() {
    const requested_tiles_buffer = ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      Name.from(requested_tiles_buf_config.name)
    );
    await requested_tiles_buffer.read(
      this.cpu_requested_tiles,
      this.cpu_requested_tiles.byteLength,
      0,
      0,
      Uint32Array
    );
  }

  #tile_request_uniforms = [];
  _setup_tile_request_uniforms(render_graph, active_tile_count) {
    this.#tile_request_uniforms.length = 0;

    if (tile_request_uniform_configs.length < active_tile_count) {
      tile_request_uniform_configs.length = active_tile_count;
      for (let i = 0; i < active_tile_count; i++) {
        tile_request_uniform_configs[i] = {
          name: `tile_request_index_uniform_${i}`,
          raw_data: new Uint32Array([i]),
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        };
      }
    }
    for (let i = 0; i < active_tile_count; i++) {
      const buffer = render_graph.create_buffer(tile_request_uniform_configs[i]);
      this.#tile_request_uniforms.push(buffer);
    }

    return this.#tile_request_uniforms;
  }

  add_debug_passes(render_graph, force_recreate, debug_view) {
    const renderer = Renderer.get();
    const image_extent = renderer.get_canvas_resolution();
    // Debug AS-VSM views
    if (debug_view === DebugDrawType.ASVSM_ShadowAtlas) {
      debug_shadow_atlas_config.width = image_extent.width;
      debug_shadow_atlas_config.height = image_extent.height;
      debug_shadow_atlas_config.force = force_recreate;
      this.debug_shadow_atlas_image = render_graph.create_image(debug_shadow_atlas_config);
      render_graph.add_pass(
        "debug_shadow_atlas_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [this.shadow_atlas],
          outputs: [this.debug_shadow_atlas_image],
          shader_setup: debug_shadow_atlas_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.draw_quad(pass);
        }
      );
    }
    if (debug_view === DebugDrawType.ASVSM_ShadowPageTable) {
      debug_page_table_config.width = page_table_config.width;
      debug_page_table_config.height = page_table_config.height;
      debug_page_table_config.force = force_recreate;
      this.debug_page_table_image = render_graph.create_image(debug_page_table_config);
      render_graph.add_pass(
        "debug_page_table_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [this.page_table, this.settings_buf],
          outputs: [this.debug_page_table_image],
          shader_setup: debug_page_table_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          // make sure we cover the full image before drawing the quad
          pass.pass.setViewport(
            0, 0,
            page_table_config.width,
            page_table_config.height,
            0, 1
          );
          MeshTaskQueue.draw_quad(pass);
        }
      );
    }
    if (debug_view === DebugDrawType.ASVSM_TileOverlay) {
      debug_tile_overlay_config.width = image_extent.width;
      debug_tile_overlay_config.height = image_extent.height;
      debug_tile_overlay_config.force = force_recreate;
      this.debug_tile_overlay_image = render_graph.create_image(debug_tile_overlay_config);

      render_graph.add_pass(
        "debug_tile_overlay_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [
            this.page_table,
            this.position_texture,
            this.settings_buf,
            this.lights_buffer,
          ],
          outputs: [this.debug_tile_overlay_image],
          shader_setup: debug_tile_overlay_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.draw_quad(pass);
        }
      );
    }
  }
}
