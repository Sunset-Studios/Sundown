import { RenderPassFlags } from "../renderer_types.js";

const gtao_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gtao.wgsl" },
  },
};

const gtao_bilateral_shader_setup = {
  pipeline_shaders: {
    compute: { path: "gi/gtao_bilateral.wgsl" },
  },
};

const ao_image_config = {
  name: "gtao_ao",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};
const ao_blur_image_config = {
  name: "gtao_ao_blur",
  format: "r32float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  clear_value: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
  force: false,
};
const bent_image_config = {
  name: "gtao_bent_normal",
  format: "rgba16float",
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};
const gtao_params_config = {
  name: "gtao_params",
  raw_data: null,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};
const gtao_bilateral_params_config = {
  name: "gtao_bilateral_params",
  raw_data: null,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  force: false,
};

export class GTAO {
  constructor(
    radius = 0.5,
    bias = 0.01,
    sample_count = 8,
    radius_bilateral = 4,
    normal_power = 32,
    sigma_ao = 0.25
  ) {
    this.radius = radius;
    this.bias = bias;
    this.sample_count = sample_count;
    this.radius_bilateral = radius_bilateral;
    this.normal_power = normal_power;
    this.sigma_ao = sigma_ao;

    this.ao_texture = null;
    this.ao_blur_texture = null;
    this.bent_normal_texture = null;
    this.gtao_params_buffer = null;
    this.gtao_bilateral_params_buffer = null;
  }

  add_passes(
    render_graph,
    { position_texture, normal_texture, width, height, force_recreate = false }
  ) {
    ao_image_config.width = width;
    ao_image_config.height = height;
    ao_image_config.force = force_recreate;

    ao_blur_image_config.width = width;
    ao_blur_image_config.height = height;
    ao_blur_image_config.force = force_recreate;

    bent_image_config.width = width;
    bent_image_config.height = height;
    bent_image_config.force = force_recreate;

    gtao_params_config.force = force_recreate;
    if (force_recreate) {
      gtao_params_config.raw_data = new Float32Array([
        this.radius,
        this.bias,
        this.sample_count,
        0.0,
      ]);
    }

    gtao_bilateral_params_config.force = force_recreate;
    if (force_recreate) {
      gtao_bilateral_params_config.raw_data = new Float32Array([
        this.radius_bilateral,
        this.normal_power,
        this.radius * 0.25,
        this.radius_bilateral * 0.5,
        this.sigma_ao,
        0.0,
      ]);
    }

    this.ao_texture = render_graph.create_image(ao_image_config);
    this.ao_blur_texture = render_graph.create_image(ao_blur_image_config);
    this.bent_normal_texture = render_graph.create_image(bent_image_config);
    this.gtao_params_buffer = render_graph.create_buffer(gtao_params_config);
    this.gtao_bilateral_params_buffer = render_graph.create_buffer(gtao_bilateral_params_config);

    render_graph.add_pass(
      "gtao",
      RenderPassFlags.Compute,
      {
        inputs: [
          position_texture,
          normal_texture,
          this.ao_texture,
          this.bent_normal_texture,
          this.gtao_params_buffer,
        ],
        outputs: [this.ao_texture, this.bent_normal_texture],
        shader_setup: gtao_shader_setup,
      },
      (g, fd, encoder) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );

    render_graph.add_pass(
      "gtao_bilateral",
      RenderPassFlags.Compute,
      {
        inputs: [
          position_texture,
          normal_texture,
          this.ao_texture,
          this.ao_blur_texture,
          this.gtao_bilateral_params_buffer,
        ],
        outputs: [this.ao_blur_texture],
        shader_setup: gtao_bilateral_shader_setup,
      },
      (g, fd, encoder) => {
        const pass = g.get_physical_pass(fd.current_pass);
        pass.dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1);
      }
    );
  }
}
