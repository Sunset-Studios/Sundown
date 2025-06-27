import { EntityFlags } from "../../core/minimal.js";
import { SharedViewBuffer } from "../../core/shared_data.js";
import { EntityManager } from "../../core/ecs/entity.js";
import { DEFAULT_CHUNK_CAPACITY } from "../../core/ecs/solar/types.js";
import { LightFragment } from "../../core/ecs/fragments/light_fragment.js";
import { Renderer } from "../renderer.js";
import { RenderPassFlags, DebugDrawType } from "../renderer_types.js";
import { MeshTaskQueue } from "../mesh_task_queue.js";
import {
  rgba8unorm_format,
  rgba16float_format,
  r32uint_format,
  depth24plus_format,
  load_op_load,
  load_op_clear,
} from "../../utility/config_permutations.js";

// ============ GPU Resource Configs ============

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

const settings_buf_config = {
  name: "shadow_settings_buf",
  raw_data: new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]),
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
};

const lru_buf_config = {
  name: "shadow_lru_buf",
  raw_data: null,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

// ========== NEW: per-light view & shadow index buffers ==========
const light_view_buf_config = {
  name: "shadow_light_view_buf",
  size: 0, // filled at runtime
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
};

const light_shadow_idx_buf_config = {
  name: "shadow_light_shadow_idx_buf",
  size: 0, // filled at runtime
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

const debug_tile_render_output_config = {
  name: "debug_tile_render_output",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
};

// Shadow atlas depth data stored in a GPUBuffer instead of a texture to access atomics
const shadow_atlas_buf_config = {
  name: "shadow_atlas_buf",
  size: 0, // filled at runtime (atlas_size^2 * pools * 4)
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
};

const dummy_depth_image_config = {
  name: "shadow_dummy_depth",
  format: depth24plus_format,
  width: 0, // filled at runtime
  height: 0, // filled at runtime
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
};

const dummy_color_image_config = {
  name: "shadow_dummy_color",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
};

const light_draw_uniform_configs = [];
// ============ Shader Setup ============

const feedback_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/feedback.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const page_table_update_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/page_table_update.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const tile_clear_shader_setup = {
  pipeline_shaders: {
    compute: { path: "shadow/as_vsm/tile_clear.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const render_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "shadow/as_vsm/tile_render.vert.wgsl", defines: { SHADOWS_ENABLED: true } },
    fragment: { path: "shadow/as_vsm/tile_render.frag.wgsl", defines: { SHADOWS_ENABLED: true } },
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

const debug_tile_render_output_shader_setup = {
  pipeline_shaders: {
    vertex: { path: "fullscreen.wgsl" },
    fragment: { path: "shadow/as_vsm/debug_tile_render.wgsl", defines: { SHADOWS_ENABLED: true } },
  },
};

const MAX_NUM_TEXTURE_POOLS = 1;

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
      this.physical_tiles_per_row * this.physical_tiles_per_row * MAX_NUM_TEXTURE_POOLS;
    this.cached_light_count = null;
    this.active_view_indices = [];
    this.active_shadow_indices = [];

    this.lights_query = EntityManager.create_query([LightFragment]);

    this.prev_grid = new Map();

    AdaptiveSparseVirtualShadowMaps.all_instances.push(this);
  }

  add_passes(
    render_graph,
    {
      position_texture,
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

    // ────────────────────────────────────────────────────────────────
    // Build per-light view & shadow index buffers (active, shadow-casting only)
    // ────────────────────────────────────────────────────────────────
    if (force_recreate) {
      this.active_view_indices.length = 0;
      this.active_shadow_indices.length = 0;

      this.lights_query.for_each_chunk((chunk, flags, counts, archetype) => {
        const lights = chunk.get_fragment_view(LightFragment);
        for (let i = 0; i < DEFAULT_CHUNK_CAPACITY; ++i) {
          const flag = flags[i];
          if (
            (flag & EntityFlags.ALIVE) === 0 ||
            lights.shadow_casting[i] === 0 ||
            lights.active[i] === 0
          ) {
            continue;
          }

          const v_idx = lights.view_index[i];
          const s_idx = lights.shadow_index[i];
          if (v_idx >= 0 && s_idx >= 0) {
            this.active_view_indices.push(v_idx);
            this.active_shadow_indices.push(s_idx);
          }
        }
      });

      light_view_buf_config.raw_data = new Uint32Array(this.active_view_indices);
      light_shadow_idx_buf_config.raw_data = new Uint32Array(this.active_shadow_indices);
    }

    // Create / resize per-light view buffer
    light_view_buf_config.force = force_recreate;
    this.light_view_buf = render_graph.create_buffer(light_view_buf_config);

    // Create / resize per-light shadow index buffer
    light_shadow_idx_buf_config.force = force_recreate;
    this.light_shadow_idx_buf = render_graph.create_buffer(light_shadow_idx_buf_config);

    // Create / resize settings buffer
    settings_buf_config.force = force_recreate;
    this.settings_buf = render_graph.create_buffer(settings_buf_config);

    // Allocate one bitmask per light so tiles are tracked independently per-light
    const bitmask_u32_stride = Math.ceil(this.total_virtual_tiles / 32); // words per light (includes all LODs)
    const total_bitmask_u32 = bitmask_u32_stride * adjusted_light_count;

    bitmask_buf_config.size = total_bitmask_u32 * 4;
    bitmask_buf_config.force = force_recreate;
    this.bitmask_buf = render_graph.create_buffer(bitmask_buf_config);
    this.bitmask_u32_stride = bitmask_u32_stride;
    this.bitmask_u32_count = total_bitmask_u32;

    // Create Page Table storage texture
    page_table_config.width = this.virtual_tiles_per_row;
    page_table_config.height = this.virtual_tiles_per_row;
    page_table_config.depth = adjusted_light_count * this.max_lods;
    page_table_config.force = force_recreate;
    this.page_table = render_graph.create_image(page_table_config);

    // Create / update the dummy render-target image
    dummy_depth_image_config.width = this.atlas_size;
    dummy_depth_image_config.height = this.atlas_size;
    dummy_depth_image_config.force = force_recreate;
    this.dummy_depth_image = render_graph.create_image(dummy_depth_image_config);

    dummy_color_image_config.width = this.atlas_size;
    dummy_color_image_config.height = this.atlas_size;
    dummy_color_image_config.force = force_recreate;
    this.dummy_color_image = render_graph.create_image(dummy_color_image_config);

    // Create storage-buffer version of the atlas for race-free depth updates
    const total_pixels = this.atlas_size * this.atlas_size * MAX_NUM_TEXTURE_POOLS;
    shadow_atlas_buf_config.size = total_pixels * Uint32Array.BYTES_PER_ELEMENT;
    shadow_atlas_buf_config.force = force_recreate;
    if (force_recreate) {
      const shadow_atlas_raw = new Uint32Array(total_pixels);
      // Doing atomic min in shader so we need to fill with max uint
      shadow_atlas_raw.fill(16777215);
      shadow_atlas_buf_config.raw_data = shadow_atlas_raw;
    }
    this.shadow_atlas_buf = render_graph.create_buffer(shadow_atlas_buf_config);

    // Create LRU ring buffer
    if (force_recreate) {
      const total_lru_entries =
        this.physical_tiles_per_row * this.physical_tiles_per_row * MAX_NUM_TEXTURE_POOLS;
      // Slot 0 is used as the head pointer for the atomic ring-buffer.
      const lru_raw = new Uint32Array(total_lru_entries + 1);
      lru_raw[0] = 0; // head pointer starts at 0
      for (let i = 0; i < total_lru_entries; i++) {
        lru_raw[i + 1] = i; // physical page id
      }
      lru_buf_config.raw_data = lru_raw;
    }
    lru_buf_config.force = force_recreate;
    this.lru_buf = render_graph.create_buffer(lru_buf_config);
    // Can discard lru_raw now that buffer is created
    lru_buf_config.raw_data = null;

    // ────────────────────────────────────────────────────────────────
    // Clear Shadow Atlas Targets
    // ────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "as_vsm_clear_shadow_atlas",
      RenderPassFlags.Graphics,
      {
        outputs: [this.dummy_depth_image, this.dummy_color_image],
        b_skip_pass_pipeline_setup: true,
        b_skip_pass_bind_group_setup: true,
      },
      (graph, frame_data, encoder) => {}
    );

    // ────────────────────────────────────────────────────────────────
    // VSM Setup Pass 
    // ────────────────────────────────────────────────────────────────
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
            MAX_NUM_TEXTURE_POOLS, // max_physical_pools
            this.clip0_extent, // clip0_extent
          ])
        );

        // Clear bitmask buffer – zero all words for every light
        const bitmask = graph.get_physical_buffer(this.bitmask_buf);
        const bitmask_raw = new Uint32Array(this.bitmask_u32_count);
        bitmask.write_raw(bitmask_raw);

        // Set dummy depth image to load_op_load
        const depth_dummy_image = graph.get_physical_image(this.dummy_depth_image);
        depth_dummy_image.config.load_op = load_op_load;
      }
    );

    // ────────────────────────────────────────────────────────────────
    // Screen-space Feedback
    // ────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "as_vsm_feedback",
      RenderPassFlags.Compute,
      {
        inputs: [
          position_texture,
          this.settings_buf,
          this.bitmask_buf,
          this.light_view_buf,
          this.light_shadow_idx_buf,
          light_count_buffer,
        ],
        outputs: [this.bitmask_buf],
        shader_setup: feedback_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const position_img = graph.get_physical_image(position_texture);
        const w = position_img.config.width;
        const h = position_img.config.height;
        const light_groups = Math.ceil(adjusted_light_count / 4);
        pass.dispatch(Math.ceil(w / 8), Math.ceil(h / 8), light_groups);
      }
    );

    // ────────────────────────────────────────────────────────────────
    // Page-table update (allocate physical pages for requested tiles)
    //   Must run BEFORE raster passes so they render into up-to-date atlas pages.
    // ────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "as_vsm_update_page_table",
      RenderPassFlags.Compute,
      {
        inputs: [
          this.lru_buf,
          this.page_table,
          this.light_shadow_idx_buf,
          this.settings_buf,
          this.bitmask_buf,
          light_count_buffer,
        ],
        outputs: [this.page_table],
        shader_setup: page_table_update_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const bitmask_groups = Math.ceil(this.bitmask_u32_stride / 8);
        const light_groups   = Math.ceil(adjusted_light_count / 4);
        pass.dispatch(bitmask_groups, bitmask_groups, light_groups);
      }
    );

    // ────────────────────────────────────────────────────────────────
    // Raster shadow casters for each requested tile
    // ────────────────────────────────────────────────────────────────
    const light_uniforms = this._setup_light_draw_uniforms(render_graph, adjusted_light_count);
    let snapped_this_frame = false;
    for (let i = 0; i < this.active_view_indices.length; i++) {
      const view_index = this.active_view_indices[i];
      const shadow_idx = this.active_shadow_indices[i];

      const light_view = SharedViewBuffer.get_view_data(view_index);
      // ------------------- snap detection -------------------
      const snap = this.clip0_extent / this.virtual_dim;   // texel size
      const gx = Math.round(light_view.view_position[0] / snap);
      const gz = Math.round(light_view.view_position[2] / snap);

      const prev = this.prev_grid.get(shadow_idx);
      if (!prev || prev.gx !== gx || prev.gz !== gz) {
        this.prev_grid.set(shadow_idx, { gx, gz });
        snapped_this_frame = true;               // some light moved ≥ 1 texel
      }

      // Use camera view visibility buffer to ensure objects are drawn
      const visible_object_instances = view_visibility_buffers[view_index];

      render_graph.add_pass(
        `as_vsm_render_light_${i}`,
        RenderPassFlags.Graphics,
        {
          inputs: [
            transforms_buffer,
            object_instances,
            visible_object_instances,
            this.settings_buf,
            this.page_table,
            light_uniforms[i],
            this.bitmask_buf,
            this.light_view_buf,
            this.light_shadow_idx_buf,
            this.shadow_atlas_buf,
          ],
          outputs: [this.shadow_atlas_buf, this.dummy_depth_image, this.dummy_color_image],
          shader_setup: render_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.submit_indexed_indirect_draws(
            pass,
            view_index,
            true, /* skip_material_bind */
            false, /* opaque_only */
            true, /* depth_only */
          );
        }
      );
    }

    // ────────────────────────────────────────────────────────────────
    // Clear newly allocated physical tiles
    // ────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "as_vsm_clear_dirty_tiles",
      RenderPassFlags.Compute,
      {
        inputs: [this.page_table, this.shadow_atlas_buf, this.settings_buf],
        outputs: [this.page_table, this.shadow_atlas_buf],
        shader_setup: tile_clear_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const pt_image = graph.get_physical_image(this.page_table);
        // Dispatch one workgroup per virtual tile in the page table.
        pass.dispatch(pt_image.config.width, pt_image.config.height, pt_image.config.depth);
      }
    );

    // ────────────────────────────────────────────────────────────────
    // VSM post-update pass 
    // ────────────────────────────────────────────────────────────────
    render_graph.add_pass(
      "as_vsm_post_update",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const depth_dummy_image = graph.get_physical_image(this.dummy_depth_image);
        depth_dummy_image.config.load_op = load_op_clear;
      }
    );

    // Debug AS-VSM views
    if (debug_view !== DebugDrawType.None) {
      this.add_debug_passes(render_graph, force_recreate, debug_view, position_texture);
    }

    page_table_config.force = force_recreate || snapped_this_frame;
    shadow_atlas_buf_config.force = force_recreate || snapped_this_frame;
    bitmask_buf_config.force = force_recreate || snapped_this_frame;
  }

  #light_draw_uniforms = [];
  _setup_light_draw_uniforms(render_graph, shadows_count) {
    this.#light_draw_uniforms.length = 0;

    // Ensure configs array is large enough and update raw_data per frame
    if (light_draw_uniform_configs.length < shadows_count) {
      light_draw_uniform_configs.length = shadows_count;
    }

    for (let i = 0; i < shadows_count; i++) {
      light_draw_uniform_configs[i] = {
        name: `light_draw_uniform_${i}`,
        raw_data: new Uint32Array([i]),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      };

      const buffer = render_graph.create_buffer(light_draw_uniform_configs[i]);
      this.#light_draw_uniforms.push(buffer);
    }

    return this.#light_draw_uniforms;
  }

  add_debug_passes(render_graph, force_recreate, debug_view, position_texture) {
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
          inputs: [this.shadow_atlas_buf, this.settings_buf],
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
          pass.pass.setViewport(0, 0, page_table_config.width, page_table_config.height, 0, 1);
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
          inputs: [this.page_table, position_texture, this.settings_buf, this.light_view_buf, this.shadow_atlas_buf],
          outputs: [this.debug_tile_overlay_image],
          shader_setup: debug_tile_overlay_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.draw_quad(pass);
        }
      );
    }
    if (debug_view === DebugDrawType.ASVSM_TileRenderOutput) {
      debug_tile_render_output_config.width = image_extent.width;
      debug_tile_render_output_config.height = image_extent.height;
      debug_tile_render_output_config.force = force_recreate;
      this.debug_tile_render_output_image = render_graph.create_image(debug_tile_render_output_config);

      render_graph.add_pass(
        "debug_tile_render_output_pass",
        RenderPassFlags.Graphics,
        {
          inputs: [this.dummy_depth_image],
          outputs: [this.debug_tile_render_output_image],
          shader_setup: debug_tile_render_output_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          MeshTaskQueue.draw_quad(pass);
        }
      );
    }
  }
}
