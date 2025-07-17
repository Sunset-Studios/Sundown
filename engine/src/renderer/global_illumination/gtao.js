import { RenderPassFlags } from "../renderer_types.js";

const gtao_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gtao.wgsl" },
  },
};

export class GTAO {
  constructor(radius = 5.0, bias = 0.01, sample_count = 8) {
    this.radius = radius;
    this.bias = bias;
    this.sample_count = sample_count;
    this.ao_texture = null;
    this.bent_normal_texture = null;
    this.params_buffer = null;
  }

  add_passes(
    render_graph,
    { depth_texture, position_texture, normal_texture, width, height, force_recreate = false }
  ) {
    const ao_image_config = {
      name: "gtao_ao",
      format: "r32float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    };
    const bent_image_config = {
      name: "gtao_bent_normal",
      format: "rgba16float",
      width: width,
      height: height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    };
    const params_config = {
      name: "gtao_params",
      raw_data: new Float32Array([this.radius, this.bias, this.sample_count, 0.0]),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    };

    this.ao_texture = render_graph.create_image(ao_image_config);
    this.bent_normal_texture = render_graph.create_image(bent_image_config);
    this.params_buffer = render_graph.create_buffer(params_config);

    render_graph.add_pass(
      "gtao",
      RenderPassFlags.Compute,
      {
        inputs: [
          depth_texture,
          position_texture,
          normal_texture,
          this.ao_texture,
          this.bent_normal_texture,
          this.params_buffer,
        ],
        outputs: [this.ao_texture, this.bent_normal_texture],
        shader_setup: gtao_shader_setup,
      },
      (g, fd, encoder) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}
