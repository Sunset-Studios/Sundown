import { MeshTaskQueue } from "../mesh_task_queue.js";
import { RenderPassFlags } from "../renderer_types.js";
import { Texture } from "../texture.js";
import { r32uint_format } from "../../utility/config_permutations.js";

const visibility_entity_image_config = {
  name: "visibility_entity",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  clear_value: { r: 0xffffffff, g: 0, b: 0, a: 0 },
  force: false,
};

const visibility_surface_image_config = {
  name: "visibility_surface",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const visibility_barycentric_image_config = {
  name: "visibility_barycentric",
  format: r32uint_format,
  width: 0,
  height: 0,
  usage:
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.TEXTURE_BINDING |
    GPUTextureUsage.COPY_SRC,
  force: false,
};

const meshlet_depth_prepass_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "visibility/meshlet_draw_standard.wgsl",
      defines: { MESHLET_DEPTH_PASS: true },
    },
    fragment: {
      path: "visibility/meshlet_draw_standard.wgsl",
      defines: { MESHLET_DEPTH_PASS: true },
    },
  },
  rasterizer_state: {
    cull_mode: "none",
  },
  depth_write_enabled: true,
  depth_stencil_compare_op: "less",
};

const meshlet_visibility_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "visibility/meshlet_draw_standard.wgsl",
      defines: { MESHLET_RASTER_PASS: true },
    },
    fragment: {
      path: "visibility/meshlet_draw_standard.wgsl",
      defines: { MESHLET_RASTER_PASS: true },
    },
  },
  rasterizer_state: {
    cull_mode: "none",
  },
  depth_write_enabled: false,
  depth_stencil_compare_op: "less-equal",
};

const visibility_gbuffer_resolve_shader_setup = {
  pipeline_shaders: {
    vertex: {
      path: "visibility/meshlet_draw_standard.wgsl",
      defines: { MESHLET_RESOLVE_PASS: true },
    },
    fragment: {
      path: "visibility/meshlet_draw_standard.wgsl",
      defines: { MESHLET_RESOLVE_PASS: true },
    },
  },
  rasterizer_state: {
    cull_mode: "none",
  },
};

const depth_prepass_pass_name = "depth_prepass";

export class VisibilityBufferPipeline {
  visibility_entity_image = null;
  visibility_surface_image = null;
  visibility_barycentric_image = null;
  registered_visibility_entity_image = null;
  registered_visibility_surface_image = null;
  registered_visibility_barycentric_image = null;

  recreate_persistent_resources(image_extent, force_recreate = false) {
    visibility_entity_image_config.width = image_extent.width;
    visibility_entity_image_config.height = image_extent.height;
    visibility_entity_image_config.force = force_recreate;

    visibility_surface_image_config.width = image_extent.width;
    visibility_surface_image_config.height = image_extent.height;
    visibility_surface_image_config.force = force_recreate;

    visibility_barycentric_image_config.width = image_extent.width;
    visibility_barycentric_image_config.height = image_extent.height;
    visibility_barycentric_image_config.force = force_recreate;

    this.visibility_entity_image = Texture.create(visibility_entity_image_config);
    this.visibility_surface_image = Texture.create(visibility_surface_image_config);
    this.visibility_barycentric_image = Texture.create(visibility_barycentric_image_config);
  }

  register_targets(render_graph) {
    this.registered_visibility_entity_image = render_graph.register_image(
      this.visibility_entity_image.config.name
    );
    this.registered_visibility_surface_image = render_graph.register_image(
      this.visibility_surface_image.config.name
    );
    this.registered_visibility_barycentric_image = render_graph.register_image(
      this.visibility_barycentric_image.config.name
    );

    return this.get_registered_targets();
  }

  get_registered_targets() {
    return {
      visibility_entity_image: this.registered_visibility_entity_image,
      visibility_surface_image: this.registered_visibility_surface_image,
      visibility_barycentric_image: this.registered_visibility_barycentric_image,
    };
  }

  add_depth_prepass(
    render_graph,
    { enabled, meshlet_draw_count, depth_image, frustum_meshlet_draw_args, inputs }
  ) {
    if (!enabled || meshlet_draw_count <= 0) {
      return depth_image;
    }

    render_graph.add_pass(
      depth_prepass_pass_name,
      RenderPassFlags.Graphics,
      {
        inputs,
        outputs: [depth_image],
        shader_setup: meshlet_depth_prepass_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.pass.drawIndirect(graph.get_physical_buffer(frustum_meshlet_draw_args).buffer, 0);
      }
    );

    return depth_image;
  }

  add_visibility_raster_pass(
    render_graph,
    {
      meshlet_draw_count,
      depth_prepass_enabled,
      current_view,
      depth_image,
      occlusion_meshlet_draw_args,
      inputs,
    }
  ) {
    if (meshlet_draw_count <= 0) {
      return this.get_registered_targets();
    }

    meshlet_visibility_shader_setup.depth_write_enabled = !depth_prepass_enabled;
    meshlet_visibility_shader_setup.depth_stencil_compare_op = depth_prepass_enabled
      ? "less-equal"
      : "less";

    render_graph.add_pass(
      `visibility_buffer_raster_view_${current_view}`,
      RenderPassFlags.Graphics,
      {
        inputs,
        outputs: [
          this.registered_visibility_entity_image,
          this.registered_visibility_surface_image,
          this.registered_visibility_barycentric_image,
          depth_image,
        ],
        shader_setup: meshlet_visibility_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        pass.pass.drawIndirect(graph.get_physical_buffer(occlusion_meshlet_draw_args).buffer, 0);
      }
    );

    return this.get_registered_targets();
  }

  add_gbuffer_resolve_pass(
    render_graph,
    {
      meshlet_draw_count,
      current_view,
      depth_image,
      inputs,
      outputs,
    }
  ) {
    if (meshlet_draw_count <= 0) {
      return outputs;
    }

    render_graph.add_pass(
      `visibility_gbuffer_resolve_view_${current_view}`,
      RenderPassFlags.Graphics,
      {
        inputs: [
          this.registered_visibility_entity_image,
          this.registered_visibility_surface_image,
          this.registered_visibility_barycentric_image,
          depth_image,
          ...inputs,
        ],
        outputs,
        shader_setup: visibility_gbuffer_resolve_shader_setup,
      },
      (graph, frame_data, encoder) => {
        const pass = graph.get_physical_pass(frame_data.current_pass);
        MeshTaskQueue.draw_quad(pass);
      }
    );

    return outputs;
  }
}
