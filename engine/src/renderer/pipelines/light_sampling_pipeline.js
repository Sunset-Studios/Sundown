import { EntityManager } from "../../core/ecs/entity.js";
import { RenderPassFlags } from "../renderer_types.js";
import { register_material_buffers, register_texture_pools } from "../render_graph_utils.js";

const COMPUTE_WORKGROUP_SIZE = 128;
const EMISSIVE_LIGHT_WORD_STRIDE = 12;
const LIGHT_LIST_HEADER_WORD_COUNT = 4;
const EMPTY_LIGHT_LIST_HEADER = new Uint32Array(LIGHT_LIST_HEADER_WORD_COUNT);

const compact_lights_shader_setup = {
  pipeline_shaders: {
    compute: { path: "system_compute/compact_lights.wgsl" },
  },
};

const compact_emissive_lights_shader_setup = {
  pipeline_shaders: {
    compute: { path: "system_compute/compact_emissive_lights.wgsl" },
  },
};

const prepare_emissive_light_instances_shader_setup = {
  pipeline_shaders: {
    compute: { path: "system_compute/prepare_emissive_light_instances.wgsl" },
  },
};

/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║                         LIGHT SAMPLING PIPELINE                             ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║ Builds the frame's shared analytic and emissive light lists once, keeping  ║
 * ║ every lighting consumer on the same compact scene-light representation.     ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */
export class LightSamplingPipeline {
  add_passes(
    render_graph,
    {
      lights,
      light_capacity,
      max_emissive_lights,
      tlas_bvh2_bounds,
      tlas_bvh_info,
      blas_directory,
      index_buffer,
      entity_transforms,
      entity_index_lookup,
      force_recreate = false,
    }
  ) {
    const dense_lights = render_graph.create_buffer({
      name: "dense_lights",
      size: Math.max(LIGHT_LIST_HEADER_WORD_COUNT, light_capacity + LIGHT_LIST_HEADER_WORD_COUNT),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const emissive_lights = render_graph.create_buffer({
      name: "emissive_lights",
      size:
        LIGHT_LIST_HEADER_WORD_COUNT +
        Math.max(1, Math.floor(max_emissive_lights)) * EMISSIVE_LIGHT_WORD_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    const material_buffers = register_material_buffers(render_graph);
    const texture_pools = register_texture_pools(render_graph);

    // Pack the two instance lookups into one binding to keep compaction within
    // 10 storage buffers, including globals. This costs one dispatch and 8 bytes
    // per primitive slot; the graph reuses the allocation until capacity grows.
    const primitive_slot_capacity = Math.floor(render_graph.get_resource_config(entity_index_lookup).size / 4);
    const emissive_light_instances = render_graph.create_buffer({
      name: "emissive_light_instances",
      size: Math.max(1, primitive_slot_capacity) * 2,
      usage: GPUBufferUsage.STORAGE,
      force: force_recreate,
    });

    render_graph.add_pass(
      "prepare_emissive_light_instances",
      RenderPassFlags.Compute,
      {
        shader_setup: prepare_emissive_light_instances_shader_setup,
        inputs: [tlas_bvh_info, entity_index_lookup, material_buffers.material_offsets_buffer, emissive_light_instances],
        outputs: [emissive_light_instances],
      },
      (graph, frame_data) => {
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(primitive_slot_capacity / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "compact_lights",
      RenderPassFlags.Compute,
      {
        shader_setup: compact_lights_shader_setup,
        inputs: [lights, dense_lights],
        outputs: [dense_lights],
      },
      (graph, frame_data) => {
        graph.get_physical_buffer(dense_lights).write_raw(EMPTY_LIGHT_LIST_HEADER, 0);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(EntityManager.get_max_rows() / COMPUTE_WORKGROUP_SIZE), 1, 1);
      }
    );

    render_graph.add_pass(
      "compact_emissive_lights",
      RenderPassFlags.Compute,
      {
        shader_setup: compact_emissive_lights_shader_setup,
        inputs: [
          tlas_bvh2_bounds,
          tlas_bvh_info,
          blas_directory,
          index_buffer,
          entity_transforms,
          material_buffers.params_gpu_buffer,
          emissive_light_instances,
          material_buffers.material_palette_buffer,
          emissive_lights,
          texture_pools.albedo,
          texture_pools.emission,
        ],
        outputs: [emissive_lights],
      },
      (graph, frame_data) => {
        const bounds_buffer = graph.get_physical_buffer(tlas_bvh2_bounds);
        graph.get_physical_buffer(emissive_lights).write_raw(EMPTY_LIGHT_LIST_HEADER, 0);
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(Math.floor(bounds_buffer.config.size / 32) / COMPUTE_WORKGROUP_SIZE),
            1,
            1
          );
      }
    );

    return { dense_lights, emissive_lights };
  }
}
