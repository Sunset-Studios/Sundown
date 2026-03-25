import {
  rgba16float_format,
  depth32float_format,
  one_one_blend_config,
} from "../utility/config_permutations.js";
import { RenderPassFlags } from "./renderer_types.js";

const main_albedo_image_config = {
  name: "main_albedo",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};

const main_smra_image_config = {
  name: "main_smra",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};

const main_normal_image_config = {
  name: "main_normal_0",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const main_normal_image2_config = {
  name: "main_normal_1",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING |
    GPUTextureUsage.COPY_DST,
  force: false,
};

const main_motion_emissive_image_config = {
  name: "main_motion_emissive",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  force: false,
};

const main_transparency_accum_image_config = {
  name: "main_transparency_accum",
  format: rgba16float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.STORAGE_BINDING,
  blend: one_one_blend_config,
  force: false,
};

const main_depth_image_config = {
  name: "main_depth_0",
  format: depth32float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const main_depth_image2_config = {
  name: "main_depth_1",
  format: depth32float_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_DST,
  force: false,
};

export class GBufferTargetsPipeline {
  create_targets(
    render_graph,
    {
      image_extent,
      force_recreate = false,
      include_prev_depth = true,
      include_prev_normal = false,
      include_transparency_accum = false,
    }
  ) {
    main_albedo_image_config.width = image_extent.width;
    main_albedo_image_config.height = image_extent.height;
    main_albedo_image_config.force = force_recreate;

    main_smra_image_config.width = image_extent.width;
    main_smra_image_config.height = image_extent.height;
    main_smra_image_config.force = force_recreate;

    main_normal_image_config.width = image_extent.width;
    main_normal_image_config.height = image_extent.height;
    main_normal_image_config.force = force_recreate;

    main_motion_emissive_image_config.width = image_extent.width;
    main_motion_emissive_image_config.height = image_extent.height;
    main_motion_emissive_image_config.force = force_recreate;

    main_depth_image_config.width = image_extent.width;
    main_depth_image_config.height = image_extent.height;
    main_depth_image_config.force = force_recreate;

    main_depth_image2_config.width = image_extent.width;
    main_depth_image2_config.height = image_extent.height;
    main_depth_image2_config.force = force_recreate;

    main_normal_image2_config.width = image_extent.width;
    main_normal_image2_config.height = image_extent.height;
    main_normal_image2_config.force = force_recreate;

    main_transparency_accum_image_config.width = image_extent.width;
    main_transparency_accum_image_config.height = image_extent.height;
    main_transparency_accum_image_config.force = force_recreate;

    const targets = {
      main_albedo_image: render_graph.create_image(main_albedo_image_config),
      main_smra_image: render_graph.create_image(main_smra_image_config),
      main_normal_image: render_graph.create_image(main_normal_image_config),
      main_motion_emissive_image: render_graph.create_image(main_motion_emissive_image_config),
      main_depth_image: render_graph.create_image(main_depth_image_config),
      prev_depth_image: include_prev_depth
        ? render_graph.create_image(main_depth_image2_config)
        : null,
      prev_normal_image: include_prev_normal
        ? render_graph.create_image(main_normal_image2_config)
        : null,
      main_transparency_accum_image: include_transparency_accum
        ? render_graph.create_image(main_transparency_accum_image_config)
        : null,
    };

    return targets;
  }

  add_clear_pass(
    render_graph,
    {
      pass_name,
      targets,
      visibility_targets = [],
    }
  ) {
    const outputs = [
      targets.main_albedo_image,
      targets.main_smra_image,
      targets.main_normal_image,
      targets.main_motion_emissive_image,
      ...visibility_targets,
    ];

    if (targets.main_transparency_accum_image) {
      outputs.push(targets.main_transparency_accum_image);
    }

    outputs.push(targets.main_depth_image);

    render_graph.add_pass(
      pass_name,
      RenderPassFlags.Graphics,
      {
        outputs,
        b_skip_pass_pipeline_setup: true,
        b_skip_pass_bind_group_setup: true,
      },
      () => {}
    );
  }

  add_set_load_op_pass(
    render_graph,
    {
      pass_name,
      targets,
      visibility_entity_image = null,
      load_op,
    }
  ) {
    render_graph.add_pass(
      pass_name,
      RenderPassFlags.GraphLocal,
      {},
      (graph, frame_data, encoder) => {
        const images = [
          graph.get_physical_image(targets.main_albedo_image),
          graph.get_physical_image(targets.main_smra_image),
          graph.get_physical_image(targets.main_normal_image),
          graph.get_physical_image(targets.main_motion_emissive_image),
          visibility_entity_image ? graph.get_physical_image(visibility_entity_image) : null,
          targets.main_transparency_accum_image
            ? graph.get_physical_image(targets.main_transparency_accum_image)
            : null,
          graph.get_physical_image(targets.main_depth_image),
        ];

        for (let i = 0; i < images.length; ++i) {
          if (images[i]) {
            images[i].config.load_op = load_op;
          }
        }
      }
    );
  }
}
