import { Typed2DFrameArray } from "../../memory/container.js";
import { Name } from "../../utility/names.js";
import { InstanceCuller } from "./instance_culler.js";
import { ResourceCache } from "../resource_cache.js";
import { RenderPassFlags, CacheTypes } from "../renderer_types.js";

const compute_dirty_movable_entities_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/dirty_movable_entities.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

const compute_dirty_slice_reducer_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/dirty_slice_reducer.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

const compute_shadow_meshlet_cull_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/cull_shadow_meshlets.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

const compute_shadow_dirty_meshlet_compact_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "shadow/as_vsm/compact_shadow_dirty_meshlets.wgsl",
      defines: { SHADOWS_ENABLED: true },
    },
  },
};

const meshlet_draw_args_reset_data = new Uint32Array([124 * 3, 0, 0, 0]);

function should_force_buffer_resize(buffer_name, required_size, force_recreate) {
  const existing_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(buffer_name));
  return force_recreate || ((existing_buffer?.config?.size ?? 0) < required_size);
}

export class ShadowCuller extends InstanceCuller {
  shadow_meshlet_lists = new Typed2DFrameArray(16, 4, Uint32Array);
  shadow_meshlet_draw_args = new Typed2DFrameArray(16, 4, Uint32Array);
  dirty_shadow_meshlet_lists = new Typed2DFrameArray(16, 4, Uint32Array);
  dirty_shadow_meshlet_draw_args = new Typed2DFrameArray(16, 4, Uint32Array);
  shadow_meshlet_param_buffers = new Typed2DFrameArray(16, 4, Uint32Array);
  registered_light_indices = new Typed2DFrameArray(16, 4, Uint32Array);

  constructor(prev_culler = null, additional_data = null) {
    super(prev_culler, additional_data);
    this.name = "shadow";
  }

  register_view(render_graph, draw_count, view_index, clipmap_index = 0, light_index = 0, force = false) {
    this.registered_views.push(view_index);
    this.registered_clipmaps.push(clipmap_index);
    this.registered_light_indices.set(view_index, clipmap_index, light_index);

    const meshlet_count = Math.max(this.additional_data.meshlet_count ?? 0, 1);
    const meshlet_list_capacity = Math.max(meshlet_count, 1);
    const required_meshlet_list_size = meshlet_list_capacity * 4 * Uint32Array.BYTES_PER_ELEMENT;

    const draw_cull_data = render_graph.create_buffer({
      name: `shadow_draw_cull_data_view_${view_index}_clipmap_${clipmap_index}`,
      raw_data: new Uint32Array([Math.max(draw_count, 1), view_index, clipmap_index]),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force,
    });
    this.cull_data_buffers.set(view_index, clipmap_index, draw_cull_data);

    const meshlet_params = render_graph.create_buffer({
      name: `shadow_meshlet_params_view_${view_index}_clipmap_${clipmap_index}`,
      raw_data: new Uint32Array([view_index, clipmap_index, light_index, meshlet_count]),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force,
    });
    this.shadow_meshlet_param_buffers.set(view_index, clipmap_index, meshlet_params);

    const raw_meshlet_list_name = `shadow_meshlet_list_view_${view_index}_clipmap_${clipmap_index}`;
    const raw_draw_args_name = `shadow_meshlet_draw_args_view_${view_index}_clipmap_${clipmap_index}`;
    const dirty_meshlet_list_name =
      `shadow_dirty_meshlet_list_view_${view_index}_clipmap_${clipmap_index}`;
    const dirty_draw_args_name =
      `shadow_dirty_meshlet_draw_args_view_${view_index}_clipmap_${clipmap_index}`;

    this.shadow_meshlet_lists.set(
      view_index,
      clipmap_index,
      render_graph.create_buffer({
        name: raw_meshlet_list_name,
        raw_data: new Uint32Array(meshlet_list_capacity * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: should_force_buffer_resize(
          raw_meshlet_list_name,
          required_meshlet_list_size,
          force
        ),
      })
    );
    this.shadow_meshlet_draw_args.set(
      view_index,
      clipmap_index,
      render_graph.create_buffer({
        name: raw_draw_args_name,
        raw_data: meshlet_draw_args_reset_data,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        force,
      })
    );
    this.dirty_shadow_meshlet_lists.set(
      view_index,
      clipmap_index,
      render_graph.create_buffer({
        name: dirty_meshlet_list_name,
        raw_data: new Uint32Array(meshlet_list_capacity * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: should_force_buffer_resize(
          dirty_meshlet_list_name,
          required_meshlet_list_size,
          force
        ),
      })
    );
    this.dirty_shadow_meshlet_draw_args.set(
      view_index,
      clipmap_index,
      render_graph.create_buffer({
        name: dirty_draw_args_name,
        raw_data: meshlet_draw_args_reset_data,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        force,
      })
    );
  }

  init_views(render_graph, draw_count) {
    const adjusted_draw_count = Math.max(draw_count, 1);
    const meshlet_count = Math.max(this.additional_data.meshlet_count ?? 0, 1);

    render_graph.add_pass(
      `${this.name}_init_views`,
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data) => {
        for (let i = 0; i < this.registered_views.length; ++i) {
          const view_index = this.registered_views.get(i);
          const clipmap_index = this.registered_clipmaps.get(i);
          const light_index = this.registered_light_indices.get(view_index, clipmap_index) ?? 0;

          const cull_data_buffer = this.cull_data_buffers.get(view_index, clipmap_index);
          const cull_data_buffer_phys = graph.get_physical_buffer(cull_data_buffer);
          if (cull_data_buffer_phys) {
            cull_data_buffer_phys.write(new Uint32Array([adjusted_draw_count, view_index, clipmap_index]));
          }

          const shadow_meshlet_param_buffer = this.shadow_meshlet_param_buffers.get(view_index, clipmap_index);
          const shadow_meshlet_param_buffer_phys = graph.get_physical_buffer(shadow_meshlet_param_buffer);
          if (shadow_meshlet_param_buffer_phys) {
            shadow_meshlet_param_buffer_phys.write(new Uint32Array([view_index, clipmap_index, light_index, meshlet_count]));
          }
        }
      }
    );
  }

  init_visibility(render_graph, draw_count) {}

  submit_cull(render_graph, draw_count, ...args) {
    this.dispatch_culling(render_graph, Math.max(draw_count, 1), ...args);
    this.last_draw_count = Math.max(draw_count, 1);
  }

  dispatch_culling(render_graph, draw_count, lights_dirtied) {
    const meshlet_count = Math.max(this.additional_data.meshlet_count ?? 0, 0);

    if (!lights_dirtied) {
      for (let i = 0; i < this.registered_views.length; ++i) {
        const view_index = this.registered_views.get(i);
        const clipmap_index = this.registered_clipmaps.get(i);
        const visible_buf_no_occlusion = this.prev_culler.get_visibility_buffer(
          view_index,
          clipmap_index
        );
        const draw_cull_data = this.cull_data_buffers.get(view_index, clipmap_index);

        render_graph.add_pass(
          `dirty_movable_entities_${view_index}_clipmap_${clipmap_index}`,
          RenderPassFlags.Compute,
          {
            shader_setup: compute_dirty_movable_entities_shader_setup,
            inputs: [
              this.additional_data.entity_transforms,
              visible_buf_no_occlusion,
              this.additional_data.object_instances,
              draw_cull_data,
              this.additional_data.vsm_settings,
              this.additional_data.entity_flags,
              this.additional_data.bitmask,
              this.additional_data.entity_index_lookup,
              this.additional_data.page_table,
              this.additional_data.page_offset,
            ],
            outputs: [this.additional_data.page_table, this.additional_data.page_offset],
          },
          (graph, frame_data) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(Math.ceil(draw_count / 256), 1, 1);
          }
        );
      }
    }

    render_graph.add_pass(
      "dirty_slice_reducer",
      RenderPassFlags.Compute,
      {
        shader_setup: compute_dirty_slice_reducer_shader_setup,
        inputs: [
          this.additional_data.vsm_settings,
          this.additional_data.page_table,
          this.additional_data.dirty_slices,
        ],
        outputs: [this.additional_data.dirty_slices],
      },
      (graph, frame_data) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        const pt_image = graph.get_physical_image(this.additional_data.page_table);
        const x_groups = Math.ceil(pt_image.config.width / 8);
        const y_groups = Math.ceil(pt_image.config.height / 8);
        const z_groups = Math.ceil(pt_image.config.depth / 4);
        pass.dispatch(x_groups, y_groups, z_groups);
      }
    );

    for (let i = 0; i < this.registered_views.length; ++i) {
      const view_index = this.registered_views.get(i);
      const clipmap_index = this.registered_clipmaps.get(i);

      const params = this.shadow_meshlet_param_buffers.get(view_index, clipmap_index);
      const shadow_meshlet_list = this.shadow_meshlet_lists.get(view_index, clipmap_index);
      const shadow_meshlet_draw_args = this.shadow_meshlet_draw_args.get(view_index, clipmap_index);
      const dirty_shadow_meshlet_list = this.dirty_shadow_meshlet_lists.get(
        view_index,
        clipmap_index
      );
      const dirty_shadow_meshlet_draw_args = this.dirty_shadow_meshlet_draw_args.get(
        view_index,
        clipmap_index
      );

      render_graph.add_pass(
        `init_shadow_meshlet_draw_args_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data) => {
          const shadow_meshlet_draw_args_phys = graph.get_physical_buffer(shadow_meshlet_draw_args);
          if (shadow_meshlet_draw_args_phys) {
            shadow_meshlet_draw_args_phys.write(meshlet_draw_args_reset_data);
          }
          const dirty_shadow_meshlet_draw_args_phys = graph.get_physical_buffer(dirty_shadow_meshlet_draw_args);
          if (dirty_shadow_meshlet_draw_args_phys) {
            dirty_shadow_meshlet_draw_args_phys.write(meshlet_draw_args_reset_data);
          }
        }
      );

      if (meshlet_count <= 0) {
        continue;
      }

      render_graph.add_pass(
        `cull_shadow_meshlets_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: compute_shadow_meshlet_cull_shader_setup,
          inputs: [
            this.additional_data.entity_transforms,
            this.additional_data.object_instances,
            this.additional_data.meshlet_instances,
            this.additional_data.entity_index_lookup,
            this.additional_data.meshlet_buffer,
            params,
            this.additional_data.vsm_settings,
            this.additional_data.dirty_slices,
            shadow_meshlet_list,
            shadow_meshlet_draw_args,
          ],
          outputs: [shadow_meshlet_list, shadow_meshlet_draw_args],
        },
        (graph, frame_data) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(meshlet_count / 128), 1, 1);
        }
      );

      render_graph.add_pass(
        `compact_shadow_dirty_meshlets_view_${view_index}_clipmap_${clipmap_index}`,
        RenderPassFlags.Compute,
        {
          shader_setup: compute_shadow_dirty_meshlet_compact_shader_setup,
          inputs: [
            this.additional_data.entity_transforms,
            this.additional_data.object_instances,
            shadow_meshlet_list,
            shadow_meshlet_draw_args,
            this.additional_data.entity_index_lookup,
            this.additional_data.meshlet_buffer,
            params,
            this.additional_data.vsm_settings,
            this.additional_data.page_table,
            dirty_shadow_meshlet_list,
            dirty_shadow_meshlet_draw_args,
          ],
          outputs: [dirty_shadow_meshlet_list, dirty_shadow_meshlet_draw_args],
        },
        (graph, frame_data) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(meshlet_count / 128), 1, 1);
        }
      );
    }
  }

  get_dirty_shadow_meshlet_list(view_index, clipmap_index) {
    return this.dirty_shadow_meshlet_lists.get(view_index, clipmap_index);
  }

  get_dirty_shadow_meshlet_draw_args(view_index, clipmap_index) {
    return this.dirty_shadow_meshlet_draw_args.get(view_index, clipmap_index);
  }
}
