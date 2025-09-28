import { RenderPassFlags } from "../renderer_types.js";
import { RayTracer } from "./raytracer.js";
import { SharedFrameInfoBuffer, SharedViewBuffer } from "../../core/shared_data.js";

const PIXEL_INFO_SIZE = 8;
const TLAS_CANDIDATES = 4;

const path_tracer_tlas_shader_setup = {
  pipeline_shaders: {
    compute: { path: "raytracing/path_trace.wgsl" },
  },
};

export class PathTracer extends RayTracer {
  params = new Uint32Array([0, 0, 0, 0]);

  constructor() {
    super();
  }

  add_passes(
    render_graph,
    width,
    height,
    max_bounces = 4,
    spp_per_frame = 1,
    position_texture = null,
    normal_texture = null,
    tlas_bvh2_bounds = null,
    tlas_bvh4_nodes = null,
    blas_atlas = null,
    entity_transforms = null,
    mesh_asset_ids = null,
    index_buffer = null,
    force_recreate = false
  ) {
    super.setup(render_graph, width, height, force_recreate);

    const view_index = SharedFrameInfoBuffer.get_view_index();
    const view_moved = SharedViewBuffer.was_moved(view_index);

    const pixel_info = render_graph.create_buffer({
      name: "pt_pixel_info",
      size: width * height * PIXEL_INFO_SIZE * TLAS_CANDIDATES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    const pt_params = render_graph.create_buffer({
      name: "pt_params",
      size: this.params.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });

    render_graph.add_pass(
      "path_trace_reset",
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const params_buffer = graph.get_physical_buffer(pt_params);
        this.params[0] = max_bounces; // max_bounces
        this.params[1] = spp_per_frame; // spp_per_frame
        this.params[2] = view_moved ? 1 : 0; // reset_accum_flag
        this.params[3] = 32; // max_spp
        params_buffer.write_raw(this.params);
      }
    );

    render_graph.add_pass(
      "path_trace",
      RenderPassFlags.Compute,
      {
        inputs: [
          pt_params,
          pixel_info,
          tlas_bvh2_bounds,
          tlas_bvh4_nodes,
          blas_atlas,
          entity_transforms,
          index_buffer,
          mesh_asset_ids,
          position_texture,
          normal_texture,
          this.output_texture,
        ],
        outputs: [pixel_info],
        shader_setup: path_tracer_tlas_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}
