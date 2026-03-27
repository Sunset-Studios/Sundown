import { SharedViewBuffer } from "../../core/shared_data.js";
import { FrustumCuller } from "../cull/frustum_culler.js";
import { OcclusionCuller } from "../cull/occlusion_culler.js";
import { ResourceCache } from "../resource_cache.js";
import { RenderPassFlags, CacheTypes } from "../renderer_types.js";
import { Texture } from "../texture.js";
import { Name } from "../../utility/names.js";
import { r32float_format } from "../../utility/config_permutations.js";

const hzb_image_config = {
  name: "hzb",
  format: r32float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  mip_levels: 0,
  b_one_view_per_mip: true,
  force: false,
};

const hzb_reduce_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/hzb_reduce.wgsl",
    },
  },
};

const meshlet_frustum_cull_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_meshlet_frustum.wgsl",
    },
  },
};

const meshlet_occlusion_cull_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/cull_meshlet_occlusion.wgsl",
    },
  },
};

const meshlet_stats_capture_shader_setup = {
  pipeline_shaders: {
    compute: {
      path: "visibility/capture_meshlet_draw_stats.wgsl",
    },
  },
};

function read_meshlet_draw_args(buffer_name) {
  const buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(buffer_name));
  const raw_data = buffer?.config?.raw_data;
  if (!raw_data || raw_data.length < 4) {
    return null;
  }

  return {
    vertex_count: Number(raw_data[0] ?? 0),
    instance_count: Number(raw_data[1] ?? 0),
    first_vertex: Number(raw_data[2] ?? 0),
    first_instance: Number(raw_data[3] ?? 0),
  };
}

function get_meshlet_stats_buffer_force(buffer_name, force_recreate) {
  const existing_buffer = ResourceCache.get().fetch(CacheTypes.BUFFER, Name.from(buffer_name));
  return (
    force_recreate ||
    !existing_buffer?.config?.cpu_readback ||
    ((existing_buffer?.config?.usage ?? 0) & GPUBufferUsage.COPY_SRC) === 0
  );
}

export class CullingPipeline {
  hzb_image = null;
  frustum_culler = null;
  occlusion_culler = null;

  constructor() {
    this.frustum_culler = new FrustumCuller(
      null,
      /* additional_data */ {
        aabb_bounds: 0,
        object_instances: 0,
        entity_index_lookup: 0,
      }
    );

    this.occlusion_culler = new OcclusionCuller(
      this.frustum_culler,
      /* additional_data */ {
        aabb_bounds: 0,
        object_instances: 0,
        main_hzb_image: 0,
        entity_index_lookup: 0,
      }
    );
  }

  reset() {
    this.frustum_culler.reset();
    this.occlusion_culler.reset();
  }

  recreate_persistent_resources(image_extent, force_recreate = false) {
    hzb_image_config.mip_levels = Math.max(
      1,
      Math.floor(Math.log2(Math.max(image_extent.width, image_extent.height))) + 1
    );
    hzb_image_config.width = image_extent.width;
    hzb_image_config.height = image_extent.height;
    hzb_image_config.force = force_recreate;

    this.hzb_image = Texture.create(hzb_image_config);
  }

  register_targets(render_graph) {
    return {
      main_hzb_image: render_graph.register_image(this.hzb_image.config.name),
    };
  }

  register_views(
    render_graph,
    { draw_count, main_hzb_image, aabb_bounds, object_instances, entity_index_lookup }
  ) {
    this.frustum_culler.additional_data.aabb_bounds = aabb_bounds;
    this.frustum_culler.additional_data.object_instances = object_instances;
    this.frustum_culler.additional_data.entity_index_lookup = entity_index_lookup;

    this.occlusion_culler.additional_data.main_hzb_image = main_hzb_image;
    this.occlusion_culler.additional_data.aabb_bounds = aabb_bounds;
    this.occlusion_culler.additional_data.object_instances = object_instances;
    this.occlusion_culler.additional_data.entity_index_lookup = entity_index_lookup;

    const total_views = SharedViewBuffer.get_view_data_count();
    for (let view_index = 0; view_index < total_views; ++view_index) {
      if (!SharedViewBuffer.is_render_active(view_index)) continue;

      const view_data = SharedViewBuffer.get_view_data(view_index);
      const clipmap_count = view_data.clipmap_count || 1;
      const occlusion_enabled = view_data.occlusion_enabled;

      for (let clipmap_index = 0; clipmap_index < clipmap_count; ++clipmap_index) {
        this.frustum_culler.register_view(render_graph, draw_count, view_index, clipmap_index);
        if (occlusion_enabled) {
          this.occlusion_culler.register_view(
            render_graph,
            draw_count,
            view_index,
            clipmap_index
          );
        }
      }
    }
  }

  add_init_view_passes(render_graph, draw_count) {
    if (draw_count <= 0) {
      return;
    }

    this.frustum_culler.init_views(render_graph, draw_count);
    this.occlusion_culler.init_views(render_graph, draw_count);
  }

  add_frustum_cull_passes(
    render_graph,
    {
      current_view,
      draw_count,
      meshlet_draw_count,
      entity_transforms,
      object_instances,
      meshlet_instances,
      entity_index_lookup,
      meshlet_buffer,
      force_recreate,
    }
  ) {
    const culling_pass_outputs = this._create_pass_resources(render_graph, {
      current_view,
      meshlet_draw_count,
      force_recreate,
    });
    const {
      frustum_meshlet_list,
      frustum_meshlet_draw_args,
      frustum_meshlet_stats,
      meshlet_list_capacity,
    } = culling_pass_outputs;

    const meshlet_frustum_params = render_graph.create_buffer({
      name: `meshlet_frustum_params_view_${current_view}`,
      raw_data: new Uint32Array([current_view, meshlet_draw_count, 0, 0]),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    if (draw_count > 0) {
      this.frustum_culler.init_visibility(render_graph, draw_count);
      this.occlusion_culler.init_visibility(render_graph, draw_count);

      render_graph.add_pass(
        `init_meshlet_draw_args_view_${current_view}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          graph
            .get_physical_buffer(frustum_meshlet_draw_args)
            .write(new Uint32Array([124 * 3, 0, 0, 0]));
          graph
            .get_physical_buffer(culling_pass_outputs.occlusion_meshlet_draw_args)
            .write(new Uint32Array([124 * 3, 0, 0, 0]));
        }
      );

      if (__DEV__ && frustum_meshlet_stats && culling_pass_outputs.occlusion_meshlet_stats) {
        render_graph.add_pass(
          `init_meshlet_stats_view_${current_view}`,
          RenderPassFlags.GraphLocal,
          {},
          (graph, frame_data, encoder) => {
            graph.get_physical_buffer(frustum_meshlet_stats).write(new Uint32Array([0, 0, 0, 0]));
            graph
              .get_physical_buffer(culling_pass_outputs.occlusion_meshlet_stats)
              .write(new Uint32Array([0, 0, 0, 0]));
          }
        );
      }

      this.frustum_culler.submit_cull(render_graph, draw_count);

      render_graph.add_pass(
        `init_meshlet_params_view_${current_view}`,
        RenderPassFlags.GraphLocal,
        {},
        (graph, frame_data, encoder) => {
          graph
            .get_physical_buffer(meshlet_frustum_params)
            .write(new Uint32Array([current_view, meshlet_draw_count, 0, 0]));
          graph
            .get_physical_buffer(culling_pass_outputs.meshlet_occlusion_params)
            .write(new Uint32Array([current_view, meshlet_list_capacity, 0, 0]));
        }
      );
    }

    if (draw_count > 0 && meshlet_draw_count > 0) {
      render_graph.add_pass(
        `meshlet_frustum_cull_view_${current_view}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            entity_transforms,
            object_instances,
            meshlet_instances,
            entity_index_lookup,
            meshlet_buffer,
            meshlet_frustum_params,
            frustum_meshlet_list,
            frustum_meshlet_draw_args,
          ],
          outputs: [frustum_meshlet_list, frustum_meshlet_draw_args],
          shader_setup: meshlet_frustum_cull_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(meshlet_draw_count / 64), 1, 1);
        }
      );

      if (__DEV__ && frustum_meshlet_stats) {
        render_graph.add_pass(
          `capture_meshlet_frustum_stats_view_${current_view}`,
          RenderPassFlags.Compute,
          {
            inputs: [frustum_meshlet_draw_args, frustum_meshlet_stats],
            outputs: [frustum_meshlet_stats],
            shader_setup: meshlet_stats_capture_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(1, 1, 1);
          }
        );
      }
    }

    return culling_pass_outputs;
  }

  add_occlusion_cull_passes(
    render_graph,
    {
      current_view,
      draw_count,
      meshlet_draw_count,
      depth_prepass_enabled,
      main_hzb_image,
      main_depth_image,
      prev_depth_image,
      entity_transforms,
      object_instances,
      entity_index_lookup,
      meshlet_buffer,
      culling_pass_outputs,
    }
  ) {
    const {
      frustum_meshlet_list,
      frustum_meshlet_draw_args,
      occlusion_meshlet_list,
      occlusion_meshlet_draw_args,
      occlusion_meshlet_stats,
      meshlet_occlusion_params,
      meshlet_list_capacity,
    } = culling_pass_outputs;

    this._add_hzb_reduce_passes(render_graph, {
      main_hzb_image,
      depth_prepass_enabled,
      main_depth_image,
      prev_depth_image,
    });

    if (draw_count > 0) {
      this.occlusion_culler.submit_cull(render_graph, draw_count);
    }

    if (meshlet_draw_count > 0) {
      render_graph.add_pass(
        `meshlet_occlusion_cull_view_${current_view}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            main_hzb_image,
            frustum_meshlet_list,
            frustum_meshlet_draw_args,
            occlusion_meshlet_list,
            occlusion_meshlet_draw_args,
            object_instances,
            entity_index_lookup,
            entity_transforms,
            meshlet_buffer,
            meshlet_occlusion_params,
          ],
          outputs: [occlusion_meshlet_list, occlusion_meshlet_draw_args],
          shader_setup: meshlet_occlusion_cull_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);
          pass.dispatch(Math.ceil(meshlet_list_capacity / 64), 1, 1);
        }
      );

      if (__DEV__ && occlusion_meshlet_stats) {
        render_graph.add_pass(
          `capture_meshlet_occlusion_stats_view_${current_view}`,
          RenderPassFlags.Compute,
          {
            inputs: [occlusion_meshlet_draw_args, occlusion_meshlet_stats],
            outputs: [occlusion_meshlet_stats],
            shader_setup: meshlet_stats_capture_shader_setup,
          },
          (graph, frame_data, encoder) => {
            const pass = graph.get_physical_pass(frame_data.current_pass);
            pass.dispatch(1, 1, 1);
          }
        );
      }
    }

    return {
      frustum_meshlet_list,
      frustum_meshlet_draw_args,
      occlusion_meshlet_list,
      occlusion_meshlet_draw_args,
    };
  }

  _create_pass_resources(render_graph, { current_view, meshlet_draw_count, force_recreate }) {
    const meshlet_list_capacity = Math.max(meshlet_draw_count, 1);
    const frustum_meshlet_list_name = `frustum_meshlet_list_view_${current_view}`;
    const occlusion_meshlet_list_name = `occlusion_meshlet_list_view_${current_view}`;
    const frustum_meshlet_draw_args_name = `frustum_meshlet_draw_args_view_${current_view}`;
    const occlusion_meshlet_draw_args_name = `occlusion_meshlet_draw_args_view_${current_view}`;
    const frustum_meshlet_stats_name = `frustum_meshlet_stats_view_${current_view}`;
    const occlusion_meshlet_stats_name = `occlusion_meshlet_stats_view_${current_view}`;
    const required_meshlet_list_size = meshlet_list_capacity * 4 * Uint32Array.BYTES_PER_ELEMENT;
    const existing_frustum_meshlet_list = ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      Name.from(frustum_meshlet_list_name)
    );
    const existing_occlusion_meshlet_list = ResourceCache.get().fetch(
      CacheTypes.BUFFER,
      Name.from(occlusion_meshlet_list_name)
    );
    const frustum_meshlet_list_force =
      force_recreate ||
      ((existing_frustum_meshlet_list?.config?.size ?? 0) < required_meshlet_list_size);
    const occlusion_meshlet_list_force =
      force_recreate ||
      ((existing_occlusion_meshlet_list?.config?.size ?? 0) < required_meshlet_list_size);

    return {
      meshlet_list_capacity,
      frustum_meshlet_list: render_graph.create_buffer({
        name: frustum_meshlet_list_name,
        raw_data: new Uint32Array(meshlet_list_capacity * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: frustum_meshlet_list_force,
      }),
      frustum_meshlet_draw_args: render_graph.create_buffer({
        name: frustum_meshlet_draw_args_name,
        raw_data: new Uint32Array([124 * 3, 0, 0, 0]),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        force: force_recreate,
      }),
      frustum_meshlet_stats: __DEV__
        ? render_graph.create_buffer({
            name: frustum_meshlet_stats_name,
            raw_data: new Uint32Array([0, 0, 0, 0]),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            cpu_readback: true,
            own_readback: true,
            force: get_meshlet_stats_buffer_force(frustum_meshlet_stats_name, force_recreate),
          })
        : null,
      occlusion_meshlet_list: render_graph.create_buffer({
        name: occlusion_meshlet_list_name,
        raw_data: new Uint32Array(meshlet_list_capacity * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: occlusion_meshlet_list_force,
      }),
      occlusion_meshlet_draw_args: render_graph.create_buffer({
        name: occlusion_meshlet_draw_args_name,
        raw_data: new Uint32Array([124 * 3, 0, 0, 0]),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
        force: force_recreate,
      }),
      occlusion_meshlet_stats: __DEV__
        ? render_graph.create_buffer({
            name: occlusion_meshlet_stats_name,
            raw_data: new Uint32Array([0, 0, 0, 0]),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            cpu_readback: true,
            own_readback: true,
            force: get_meshlet_stats_buffer_force(occlusion_meshlet_stats_name, force_recreate),
          })
        : null,
      meshlet_occlusion_params: render_graph.create_buffer({
        name: `meshlet_occlusion_params_view_${current_view}`,
        raw_data: new Uint32Array([current_view, meshlet_list_capacity, 0, 0]),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  _add_hzb_reduce_passes(
    render_graph,
    { main_hzb_image, depth_prepass_enabled, main_depth_image, prev_depth_image }
  ) {
    const hzb_depth_image = depth_prepass_enabled ? main_depth_image : prev_depth_image;
    const hzb_params_chain = [];

    for (let i = 0; i < this.hzb_image.config.mip_levels; i++) {
      hzb_params_chain.push(
        render_graph.create_buffer({
          name: `hzb_params_${i}`,
          data: [0.0, 0.0, 0.0, 0.0],
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
      );
    }

    for (let i = 0; i < this.hzb_image.config.mip_levels; i++) {
      const src_index = i === 0 ? 0 : i - 1;
      const dst_index = i;

      render_graph.add_pass(
        `reduce_hzb_${i}`,
        RenderPassFlags.Compute,
        {
          inputs: [
            i === 0 ? hzb_depth_image : main_hzb_image,
            main_hzb_image,
            hzb_params_chain[dst_index],
          ],
          outputs: [main_hzb_image],
          input_views: [i === 0 ? 0 : i, i + 1],
          shader_setup: hzb_reduce_shader_setup,
        },
        (graph, frame_data, encoder) => {
          const pass = graph.get_physical_pass(frame_data.current_pass);

          const depth = graph.get_physical_image(hzb_depth_image);
          const hzb = graph.get_physical_image(main_hzb_image);
          const hzb_params = graph.get_physical_buffer(hzb_params_chain[dst_index]);

          const src_mip_width = Math.max(
            1,
            i === 0 ? depth.config.width : hzb.config.width >> src_index
          );
          const src_mip_height = Math.max(
            1,
            i === 0 ? depth.config.height : hzb.config.height >> src_index
          );

          const dst_mip_width = Math.max(1, hzb.config.width >> dst_index);
          const dst_mip_height = Math.max(1, hzb.config.height >> dst_index);

          hzb_params.write([src_mip_width, src_mip_height, dst_mip_width, dst_mip_height]);

          pass.dispatch((dst_mip_width + 7) / 8, (dst_mip_height + 7) / 8, 1);
        }
      );
    }
  }

  get_frustum_visibility_buffer(view_index, clipmap_index = 0) {
    return this.frustum_culler.get_visibility_buffer(view_index, clipmap_index);
  }

  get_occlusion_visibility_buffer(view_index, clipmap_index = 0) {
    return this.occlusion_culler.get_visibility_buffer(view_index, clipmap_index);
  }

  get_frustum_culler() {
    return this.frustum_culler;
  }

  get_hzb_mip_level_count() {
    return this.hzb_image?.config?.mip_levels ?? 0;
  }

  get_meshlet_stats(view_index) {
    if (!__DEV__) {
      return null;
    }

    return {
      view_index,
      frustum: read_meshlet_draw_args(`frustum_meshlet_stats_view_${view_index}`),
      occlusion: read_meshlet_draw_args(`occlusion_meshlet_stats_view_${view_index}`),
    };
  }
}
