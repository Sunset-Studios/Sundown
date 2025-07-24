import { rgba16float_format } from "../../utility/config_permutations.js";

const gi_texture_config = {
  name: "gi_image",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  force: false,
};

class GI {
  constructor() { }

  add_passes(render_graph, width, height, force_recreate = false) {
    gi_texture_config.width = width;
    gi_texture_config.height = height;
    gi_texture_config.force = force_recreate;
    this.final_gi_texture = render_graph.create_image(gi_texture_config);
  }
}

export { GI };
