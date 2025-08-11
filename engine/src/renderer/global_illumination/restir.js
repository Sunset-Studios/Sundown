import { RenderPassFlags } from "../renderer_types.js";
import { RayTracer } from "../raytracing/raytracer.js";

const restir_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/restir.wgsl" },
  },
};

const restir_output_config = {
  name: "restir_gi",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

export class ReSTIRGI extends RayTracer {
  constructor() {
    super();
  }

  add_passes(render_graph, { width, height, force_recreate = false }) {
    restir_output_config.width = width;
    restir_output_config.height = height;
    restir_output_config.force = force_recreate;

    this.output_texture = render_graph.create_image(restir_output_config);

    render_graph.add_pass(
      "restir_gi",
      RenderPassFlags.Compute,
      {
        inputs: [this.output_texture],
        outputs: [this.output_texture],
        shader_setup: restir_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}