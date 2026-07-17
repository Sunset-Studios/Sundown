import { GIAccumulator } from "./gi_pipeline.js";
import { GIRadianceRepresentation } from "./shading_strategies.js";

const shader = (path) => ({ pipeline_shaders: { compute: { path } } });

export class ProbeSHAccumulator extends GIAccumulator {
  constructor(options = {}) {
    super({
      name: "probe-sh-accumulator",
      representation: "probe-sh",
      accepted_radiance_representations: [GIRadianceRepresentation.PROBE_SH],
      shader_setups: {
        accumulate: shader("gi/ddgi_sh_probe_accumulate.wgsl"),
        depth_update: shader("gi/ddgi_depth_update.wgsl"),
        sample: shader("gi/ddgi_sh_probe_sample.wgsl"),
        resolve: shader("gi/ddgi_diffuse_resolve.wgsl"),
        atrous: shader("gi/ddgi_atrous_diffuse.wgsl"),
        ...options.shader_setups,
      },
    });
  }

  setup(render_graph, context) {
    const { width, height, probe_count, total_depth_texel_count, force_recreate } = context;
    const create_buffer = (
      semantic,
      name,
      size,
      usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    ) => this.create_buffer(render_graph, semantic, { name, size, usage, force: force_recreate });
    const create_image = (semantic, name, image_width = width, image_height = height) =>
      this.create_image(render_graph, semantic, {
        name,
        format: "rgba16float",
        width: image_width,
        height: image_height,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        force: force_recreate,
      });
    create_buffer("history_valid", "probe_sh_history_valid", probe_count, GPUBufferUsage.STORAGE);
    create_buffer("depth_moments", "probe_sh_depth_moments", total_depth_texel_count);
    create_buffer("sh_probes", "probe_sh_coefficients", probe_count * 6);
    create_buffer("msme_stats", "probe_sh_msme_stats", probe_count * 36);
    create_image("direct_output", "probe_sh_direct_output");
    create_image("diffuse_output", "probe_sh_diffuse_output");
    create_image("specular_output", "probe_sh_specular_output");
    const upscale = Math.max(1, Math.floor(context.config.diffuse_sample_upscale_factor || 1));
    context.diffuse_sample_upscale_factor = upscale;
    context.diffuse_sample_width = Math.max(1, Math.ceil(width / upscale));
    context.diffuse_sample_height = Math.max(1, Math.ceil(height / upscale));
    if (upscale > 1) {
      create_image(
        "diffuse_sample_output",
        "probe_sh_diffuse_sample_intermediate",
        context.diffuse_sample_width,
        context.diffuse_sample_height
      );
    } else {
      this.import_resource("diffuse_sample_output", this.get_resource("diffuse_output"));
    }
    create_image("atrous_ping", "probe_sh_diffuse_atrous_ping");
    create_image("atrous_pong", "probe_sh_diffuse_atrous_pong");
    this.atrous_params_data = new Float32Array([1, 0.04, 64, 1]);
    this.create_buffer(render_graph, "atrous_params", {
      name: "probe_sh_diffuse_atrous_params",
      size: this.atrous_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  record(render_graph, context, branch) {
    const trace = branch.trace_hit_cache;
    const params = trace.get_resource("params");
    const states = trace.get_resource("probe_states");
    const ray_hits = trace.get_resource("ray_hits");
    const history_valid = this.get_resource("history_valid");
    const sh_probes = this.get_resource("sh_probes");
    const msme_stats = this.get_resource("msme_stats");
    this.add_compute_pass(
      render_graph,
      "accumulate",
      "probe_sh_accumulate",
      {
        inputs: [
          params,
          trace.get_resource("update_indices"),
          ray_hits,
          history_valid,
          sh_probes,
          states,
          msme_stats,
          trace.get_resource("counters"),
        ],
        outputs: [history_valid, sh_probes, states, msme_stats],
      },
      (graph, frame_data) =>
        graph.get_physical_pass(frame_data.current_pass).dispatch(context.probes_per_frame, 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "depth_update",
      "probe_sh_depth_moments_update",
      {
        inputs: [params, ray_hits, history_valid, this.get_resource("depth_moments")],
        outputs: [this.get_resource("depth_moments")],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(context.probes_per_frame / 8),
            Math.ceil(context.max_depth_texel_count_per_probe / 8),
            1
          )
    );
  }

  record_resolve(render_graph, context, branch) {
    const trace = branch.trace_hit_cache;
    const lighting = branch.shading_strategy.scene_lighting_data;
    const inputs = context.inputs;
    const sample_output = this.get_resource("diffuse_sample_output");
    this.add_compute_pass(
      render_graph,
      "sample",
      `probe_sh_sample_${context.ping_pong_frame}`,
      {
        inputs: [
          trace.get_resource("params"),
          this.get_resource("sh_probes"),
          trace.get_resource("probe_states"),
          this.get_resource("depth_moments"),
          inputs.hzb_texture,
          inputs.gbuffer_normal,
          sample_output,
          lighting.scene_lighting_buffer,
          lighting.skybox_image,
        ],
        outputs: [sample_output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(
            Math.ceil(context.diffuse_sample_width / 16),
            Math.ceil(context.diffuse_sample_height / 16),
            1
          )
    );
    const diffuse_output = this.get_resource("diffuse_output");
    if (context.diffuse_sample_upscale_factor > 1) {
      this.add_compute_pass(
        render_graph,
        "resolve",
        `probe_sh_resolve_${context.ping_pong_frame}`,
        {
          inputs: [sample_output, inputs.hzb_texture, inputs.gbuffer_normal, diffuse_output],
          outputs: [diffuse_output],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
    }

    let final_diffuse = diffuse_output;
    let read = diffuse_output;
    let write = this.get_resource("atrous_ping");
    const pass_count =
      context.config.diffuse_atrous_enabled === false
        ? 0
        : Math.max(0, Math.floor(context.config.diffuse_atrous_pass_count || 0));
    for (let pass_index = 0; pass_index < pass_count; pass_index++) {
      this.add_graph_local_pass(
        render_graph,
        `probe_sh_atrous_upload_params_${context.ping_pong_frame}_${pass_index}`,
        (graph) => {
          this.atrous_params_data[0] = Math.pow(2, pass_index);
          this.atrous_params_data[1] = Math.max(
            0.0001,
            context.config.diffuse_atrous_phi_depth || 0.04
          );
          this.atrous_params_data[2] = Math.max(1, context.config.diffuse_atrous_phi_normal || 64);
          this.atrous_params_data[3] = Math.max(
            0.0001,
            context.config.diffuse_atrous_luma_sigma || 1
          );
          graph
            .get_physical_buffer(this.get_resource("atrous_params"))
            .write_raw(this.atrous_params_data);
        }
      );
      this.add_compute_pass(
        render_graph,
        "atrous",
        `probe_sh_atrous_${context.ping_pong_frame}_${pass_index}`,
        {
          inputs: [
            this.get_resource("atrous_params"),
            read,
            inputs.depth_texture,
            inputs.gbuffer_normal,
            write,
          ],
          outputs: [write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
      final_diffuse = write;
      read = write;
      write =
        write === this.get_resource("atrous_ping")
          ? this.get_resource("atrous_pong")
          : this.get_resource("atrous_ping");
    }
    this.import_resource("diffuse_output", final_diffuse);
  }
}

export class SurfaceCacheSHAccumulator extends GIAccumulator {
  constructor({ shader_setups = {} } = {}) {
    super({
      name: "surface-cache-sh-accumulator",
      representation: "surface-cache-sh",
      accepted_radiance_representations: [GIRadianceRepresentation.SURFACE_SH],
      shader_setups: {
        accumulate: shader("gi/surface_cache_accumulate.wgsl"),
        filter: shader("gi/surface_cache_filter.wgsl"),
        resolve: shader("gi/surface_cache_resolve.wgsl"),
        ...shader_setups,
      },
    });
  }

  setup(render_graph, context, branch) {
    const { width, height, total_patches, force_recreate } = context;
    const image_config = (name) => ({
      name,
      format: "rgba16float",
      width,
      height,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      force: force_recreate,
    });
    const sh_size = total_patches * 6;

    this.create_image(render_graph, "direct_output", image_config("surface_cache_direct_output"));
    this.create_image(render_graph, "diffuse_output", image_config("surface_cache_diffuse_output"));
    this.create_image(
      render_graph,
      "specular_output",
      image_config("surface_cache_specular_output")
    );
    this.create_buffer(render_graph, "surface_cache_sh", {
      name: "surface_cache_sh",
      size: sh_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
    this.create_buffer(render_graph, "surface_cache_sh_filtered", {
      name: "surface_cache_sh_filtered",
      size: sh_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  record(render_graph, context, branch) {
    const trace = branch.trace_hit_cache;
    const shade = branch.shading_strategy;
    const params = trace.get_resource("params");
    const surface_cache = trace.get_resource("surface_cache");
    const active_indices = trace.get_resource("active_indices");
    const counters = trace.get_resource("counters");
    const hit_info = trace.get_resource("hit_info");
    const radiance_info = shade.get_resource("radiance_info");
    const sh = this.get_resource("surface_cache_sh");
    const sh_filtered = this.get_resource("surface_cache_sh_filtered");
    const direct = this.get_resource("direct_output");
    const diffuse = this.get_resource("diffuse_output");
    const specular = this.get_resource("specular_output");
    const { total_patches, width, height, inputs } = context;

    this.add_compute_pass(
      render_graph,
      "accumulate",
      "surface_cache_sh_accumulate",
      {
        inputs: [params, surface_cache, sh, active_indices, counters, hit_info, radiance_info],
        outputs: [surface_cache, sh],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(total_patches / 128), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "filter",
      "surface_cache_sh_filter",
      {
        inputs: [params, surface_cache, sh, sh_filtered, active_indices, counters],
        outputs: [sh_filtered],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(total_patches / 128), 1, 1)
    );
    this.add_compute_pass(
      render_graph,
      "resolve",
      "surface_cache_sh_resolve",
      {
        inputs: [
          params,
          surface_cache,
          sh_filtered,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          direct,
          diffuse,
          specular,
        ],
        outputs: [direct, diffuse, specular],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(width / 8), Math.ceil(height / 8), 1)
    );
  }
}

export class PerPixelRGBAccumulator extends GIAccumulator {
  constructor(options = {}) {
    super({
      name: "per-pixel-weighted-rgb-accumulator",
      representation: "per-pixel-rgb",
      accepted_radiance_representations: [GIRadianceRepresentation.PIXEL_RGB],
      shader_setups: {
        temporal: shader("gi/pixel_temporal_reservoir.wgsl"),
        spatial_wide: shader("gi/pixel_spatial_reservoir_wide.wgsl"),
        spatial_narrow: shader("gi/pixel_spatial_reservoir_narrow.wgsl"),
        accumulate: shader("gi/pixel_accumulate.wgsl"),
        resolve: shader("gi/pixel_upscale_final.wgsl"),
        atrous: shader("gi/ddgi_atrous_diffuse.wgsl"),
        ...options.shader_setups,
      },
    });
  }

  setup(render_graph, context) {
    const { width, height, gi_width, gi_height, force_recreate } = context;
    const buffer = (semantic, name, size) =>
      this.create_buffer(render_graph, semantic, {
        name,
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        force: force_recreate,
      });
    const image = (semantic, name, image_width, image_height) =>
      this.create_image(render_graph, semantic, {
        name,
        format: "rgba16float",
        width: image_width,
        height: image_height,
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        force: force_recreate,
      });
    const reservoir_size = gi_width * gi_height * 28;
    buffer("temporal_reservoir_0", "gi_temporal_reservoir_0", reservoir_size);
    buffer("temporal_reservoir_1", "gi_temporal_reservoir_1", reservoir_size);
    buffer("spatial_reservoir_0", "gi_spatial_reservoir_0", reservoir_size);
    buffer("spatial_reservoir_1", "gi_spatial_reservoir_1", reservoir_size);
    buffer("spatial_reservoir_stage", "gi_spatial_reservoir_stage", reservoir_size);
    image("radiance_direct_0", "gi_low_radiance_direct_0", gi_width, gi_height);
    image("radiance_direct_1", "gi_low_radiance_direct_1", gi_width, gi_height);
    image("radiance_diffuse_0", "gi_low_radiance_indirect_diffuse_0", gi_width, gi_height);
    image("radiance_diffuse_1", "gi_low_radiance_indirect_diffuse_1", gi_width, gi_height);
    image("radiance_specular_0", "gi_low_radiance_indirect_specular_0", gi_width, gi_height);
    image("radiance_specular_1", "gi_low_radiance_indirect_specular_1", gi_width, gi_height);
    image("direct_output", "pixel_rgb_direct_output", width, height);
    image("diffuse_output", "pixel_rgb_diffuse_output", width, height);
    image("specular_output", "pixel_rgb_specular_output", width, height);
    image("atrous_ping", "gi_diffuse_atrous_ping", width, height);
    image("atrous_pong", "gi_diffuse_atrous_pong", width, height);
    this.atrous_params_data = new Float32Array([1, 0.04, 64.0, 1.0]);
    this.create_buffer(render_graph, "atrous_params", {
      name: "gi_diffuse_atrous_params",
      size: this.atrous_params_data.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      force: force_recreate,
    });
  }

  record(render_graph, context, branch) {
    const trace = branch.trace_hit_cache;
    const inputs = context.inputs;
    const params = trace.get_resource("params");
    const path_state = trace.get_resource("path_state");
    const frame = context.ping_pong_frame;
    const temporal_prev = this.get_resource(`temporal_reservoir_${frame}`);
    const temporal_curr = this.get_resource(`temporal_reservoir_${1 - frame}`);
    const spatial_curr = this.get_resource(`spatial_reservoir_${1 - frame}`);
    const spatial_stage = this.get_resource("spatial_reservoir_stage");
    const previous = {
      direct: this.get_resource(`radiance_direct_${frame}`),
      diffuse: this.get_resource(`radiance_diffuse_${frame}`),
      specular: this.get_resource(`radiance_specular_${frame}`),
    };
    const current = {
      direct: this.get_resource(`radiance_direct_${1 - frame}`),
      diffuse: this.get_resource(`radiance_diffuse_${1 - frame}`),
      specular: this.get_resource(`radiance_specular_${1 - frame}`),
    };
    const dispatch_2d = (semantic, name, pass_inputs, outputs, divisor) =>
      this.add_compute_pass(
        render_graph,
        semantic,
        name,
        { inputs: pass_inputs, outputs },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(
              Math.ceil(context.gi_width / divisor),
              Math.ceil(context.gi_height / divisor),
              1
            )
      );

    dispatch_2d(
      "temporal",
      `pixel_rgb_temporal_reservoir_${frame}`,
      [
        params,
        path_state,
        temporal_prev,
        temporal_curr,
        inputs.depth_texture,
        inputs.prev_depth_texture,
        inputs.gbuffer_normal,
        inputs.gbuffer_motion_emissive,
        inputs.gbuffer_normal_prev,
      ],
      [temporal_curr],
      16
    );
    dispatch_2d(
      "spatial_wide",
      `pixel_rgb_spatial_reservoir_wide_${frame}`,
      [
        params,
        temporal_curr,
        spatial_stage,
        inputs.depth_texture,
        inputs.gbuffer_normal,
        inputs.gbuffer_smra,
      ],
      [spatial_stage],
      16
    );
    dispatch_2d(
      "spatial_narrow",
      `pixel_rgb_spatial_reservoir_narrow_${frame}`,
      [
        params,
        spatial_stage,
        spatial_curr,
        inputs.depth_texture,
        inputs.gbuffer_normal,
        inputs.gbuffer_smra,
      ],
      [spatial_curr],
      16
    );
    dispatch_2d(
      "accumulate",
      `pixel_rgb_accumulate_${frame}`,
      [
        params,
        spatial_curr,
        previous.direct,
        previous.diffuse,
        previous.specular,
        inputs.depth_texture,
        inputs.prev_depth_texture,
        inputs.gbuffer_normal,
        inputs.gbuffer_normal_prev,
        inputs.gbuffer_motion_emissive,
        current.direct,
        current.diffuse,
        current.specular,
      ],
      [current.direct, current.diffuse, current.specular],
      8
    );

    const direct_output = this.get_resource("direct_output");
    const diffuse_output = this.get_resource("diffuse_output");
    const specular_output = this.get_resource("specular_output");
    this.add_compute_pass(
      render_graph,
      "resolve",
      `pixel_rgb_resolve_${frame}`,
      {
        inputs: [
          params,
          current.direct,
          current.diffuse,
          current.specular,
          inputs.depth_texture,
          inputs.gbuffer_normal,
          inputs.gbuffer_smra,
          direct_output,
          diffuse_output,
          specular_output,
        ],
        outputs: [direct_output, diffuse_output, specular_output],
      },
      (graph, frame_data) =>
        graph
          .get_physical_pass(frame_data.current_pass)
          .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
    );

    let final_diffuse = diffuse_output;
    let read = diffuse_output;
    let write = this.get_resource("atrous_ping");
    const pass_count =
      context.config.diffuse_atrous_enabled === false
        ? 0
        : Math.max(0, Math.floor(context.config.diffuse_atrous_pass_count || 0));
    for (let pass_index = 0; pass_index < pass_count; pass_index++) {
      this.add_graph_local_pass(
        render_graph,
        `pixel_rgb_atrous_upload_params_${frame}_${pass_index}`,
        (graph) => {
          this.atrous_params_data[0] = Math.pow(2, pass_index);
          this.atrous_params_data[1] = Math.max(
            0.0001,
            context.config.diffuse_atrous_phi_depth || 0.04
          );
          this.atrous_params_data[2] = Math.max(1, context.config.diffuse_atrous_phi_normal || 64);
          this.atrous_params_data[3] = Math.max(
            0.0001,
            context.config.diffuse_atrous_luma_sigma || 1
          );
          graph
            .get_physical_buffer(this.get_resource("atrous_params"))
            .write_raw(this.atrous_params_data);
        }
      );
      this.add_compute_pass(
        render_graph,
        "atrous",
        `pixel_rgb_atrous_${frame}_${pass_index}`,
        {
          inputs: [
            this.get_resource("atrous_params"),
            read,
            inputs.depth_texture,
            inputs.gbuffer_normal,
            write,
          ],
          outputs: [write],
        },
        (graph, frame_data) =>
          graph
            .get_physical_pass(frame_data.current_pass)
            .dispatch(Math.ceil(context.width / 8), Math.ceil(context.height / 8), 1)
      );
      final_diffuse = write;
      read = write;
      write =
        write === this.get_resource("atrous_ping")
          ? this.get_resource("atrous_pong")
          : this.get_resource("atrous_ping");
    }
    this.import_resource("diffuse_output", final_diffuse);
  }
}
