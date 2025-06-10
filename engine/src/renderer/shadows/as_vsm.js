import { EntityManager } from "../../core/ecs/entity.js";
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

const histogram_buf_config = {
  name: "shadow_histogram_buf",
  size: 0,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

const tile_request_uniform_configs = [];
// ============ Shader Setup ============

const histogram_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/histogram.wgsl" },
  },
};

const prefix_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/split_depth_sum.wgsl" },
  },
};

const feedback_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/feedback.wgsl" },
  },
};

const gather_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/gather.wgsl" },
  },
};

const page_table_update_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/page_table_update.wgsl" },
  },
};

const render_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "shadow/as_vsm/tile_render.vert.wgsl" },
  },
};

const debug_shadow_atlas_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "shadow/as_vsm/debug_shadow_atlas.wgsl" },
  },
};

const debug_page_table_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "shadow/as_vsm/debug_page_table.wgsl" },
  },
};

const HISTOGRAM_BINS = 64;
const MAX_TILE_REQUESTS_PER_VIEW = 64;

/**
 * Adaptive Sparse Virtual Shadow Maps (AS-VSM)
 * Scaffolding: allocates GPU resources and adds stub passes.
 */
export class AdaptiveSparseVirtualShadowMaps {
  constructor({ atlas_size, tile_size, virtual_dim, max_lods }) {
    this.tile_size = tile_size;
    this.virtual_dim = virtual_dim;
    this.atlas_size = atlas_size;
    this.max_lods = max_lods;
    this.virtual_tiles_per_row = Math.ceil(this.virtual_dim / this.tile_size);
    this.total_virtual_tiles =
      this.virtual_tiles_per_row * this.virtual_tiles_per_row * this.max_lods;
    this.physical_tiles_per_row = Math.ceil(this.atlas_size / this.tile_size);
    this.total_physical_tiles =
      this.physical_tiles_per_row * this.physical_tiles_per_row * this.max_lods;
    this.lights_query = EntityManager.create_query([LightFragment]);
    this.cached_light_count = null;

    const num_elements = 1 + MAX_TILE_REQUESTS_PER_VIEW * 3;
    this.cpu_requested_tiles = new Uint32Array(num_elements);
  }

  add_passes(
    render_graph,
    {
      depth_texture,
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

    // Create Histogram buffer
    histogram_buf_config.size = HISTOGRAM_BINS * 4;
    histogram_buf_config.force = force_recreate;
    this.histogram_buf = render_graph.create_buffer(histogram_buf_config);

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
    atlas_config.depth = adjusted_light_count;
    atlas_config.force = force_recreate;
    atlas_config.b_one_view_per_layer = true;
    this.shadow_atlas = render_graph.create_image(atlas_config);

    // Create Page Table storage texture
    page_table_config.width = this.virtual_tiles_per_row;
    page_table_config.height = this.virtual_tiles_per_row;
    page_table_config.depth = adjusted_light_count;
    page_table_config.force = force_recreate;
    this.page_table = render_graph.create_image(page_table_config);

    // Create Physical to Virtual map buffer
    const physical_tiles_per_view =
      this.physical_tiles_per_row * this.physical_tiles_per_row * this.max_lods;
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

    // Stage 0: Clear buffers
    render_graph.add_pass(
      "as_vsm_init",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        // Clear histogram buffer
        const histogram = graph.get_physical_buffer(this.histogram_buf);
        histogram.write_raw(new Float32Array(HISTOGRAM_BINS * 4));

        // Clear settings buffer
        const settings = graph.get_physical_buffer(this.settings_buf);
        settings.write_raw(
          new Float32Array([
            0, // split_depth
            this.tile_size, // tile_size
            this.virtual_dim, // virtual_dim
            this.virtual_tiles_per_row, // virtual_tiles_per_row
            this.atlas_size, // atlas_size
            this.physical_tiles_per_row, // physical_tiles_per_row
            this.max_lods, // max_lod
            this.max_tile_requests, // max_tile_requests
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

    // Stage A: Depth Histogram
    render_graph.add_pass(
      "as_vsm_histogram",
      RenderPassFlags.Compute,
      {
        inputs: [depth_texture, this.histogram_buf],
        outputs: [this.histogram_buf],
        shader_setup: histogram_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const w = this.virtual_tiles_per_row * this.max_lods;
        const h = this.physical_tiles_per_row * this.max_lods;
        pass.dispatch(Math.ceil(w / 16), Math.ceil(h / 16), 1);
      }
    );

    // Stage A.5: Prefix-sum → compute split_depth (writes to settings_buf)
    render_graph.add_pass(
      "as_vsm_split_depth_sum",
      RenderPassFlags.Compute,
      {
        inputs: [this.histogram_buf, this.settings_buf],
        outputs: [this.settings_buf],
        shader_setup: prefix_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(1, 1, 1);
      }
    );

    // Stage B: Screen-space Feedback
    render_graph.add_pass(
      "as_vsm_feedback",
      RenderPassFlags.Compute,
      {
        inputs: [depth_texture, this.settings_buf, this.bitmask_buf],
        outputs: [this.bitmask_buf],
        shader_setup: feedback_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const depth_img = graph.get_physical_image(depth_texture);
        const w = depth_img.config.width;
        const h = depth_img.config.height;
        pass.dispatch(Math.ceil(w / 8), Math.ceil(h / 8), 1);
      }
    );

    // Stage C: New Tile Gather
    render_graph.add_pass(
      "as_vsm_gather",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.bitmask_buf,
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

    // Stage D: Update Page Table
    render_graph.add_pass(
      "as_vsm_update_page_table",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.requested_tiles_buf,
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
        const light_count_groups = Math.ceil(adjusted_light_count / 4);
        pass.dispatch(requested_tiles_groups, light_count_groups, 1);
      }
    );

    // Stage E: Process active tile requests and submit draw calls per request
    const active_tile_count = this.cpu_requested_tiles[0];
    const tile_request_uniforms = this._setup_tile_request_uniforms(
      render_graph,
      active_tile_count
    );
    for (let i = 0; i < active_tile_count; i++) {
      // For simplicity, we set view index to 0. In a complete implementation, derive this from the tile request data.
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
          shader_setup: render_shader_setup, // ensure shader reads the tile index from the uniform
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.submit_indexed_indirect_draws(
            pass,
            view_index, /* view_index */
            true, /* skip_material_bind */
          );
        }
      );
    }
    
    // Debug AS-VSM views
    if (debug_view !== DebugDrawType.None) {
      this.add_debug_passes(render_graph, force_recreate, debug_view);
    }

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
      debug_page_table_config.width = image_extent.width;
      debug_page_table_config.height = image_extent.height;
      debug_page_table_config.force = force_recreate;
      this.debug_page_table_image = render_graph.create_image(debug_page_table_config);
      render_graph.add_pass(
        "debug_page_table_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [this.page_table],
          outputs: [this.debug_page_table_image],
          shader_setup: debug_page_table_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.draw_quad(pass);
        }
      );
    }
  }
}
