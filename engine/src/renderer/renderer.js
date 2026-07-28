import { MAX_BUFFERED_FRAMES } from "../core/minimal.js";
import {
  DebugDrawType,
  RenderStrategyType,
  GIStrategyType,
  AOStrategyType,
  ReflectionStrategyType,
} from "./renderer_types.js";
import { CVarSystem } from "../core/cvar_system.js";
import { DeferredShadingStrategy } from "./strategies/deferred_shading.js";
import { PathTracingStrategy } from "./strategies/path_tracing.js";
import { RenderGraph } from "./render_graph.js";
import { Texture, TextureSampler } from "./texture.js";
import { Mesh } from "./mesh.js";
import { ShaderArchive } from "./shader_archive.js";
import {
  SharedViewBuffer,
  SharedFrameInfoBuffer,
} from "../core/shared_data.js";
import { MeshData } from "./mesh_data.js";
import { global_dispatcher } from "../core/dispatcher.js";
import { profile_scope } from "../utility/performance.js";
import ExecutionQueue from "../utility/execution_queue.js";
import { GPUTimeQuery } from "./query.js";
import { log, error } from "../utility/logging.js";
import { vec2 } from "gl-matrix";
import { EngineCVars } from "../../config/cvars.js";
import { reset_ui, flush_ui, panel, label, image, UIContext } from "../ui/2d/immediate.js";
import { reset_ui_3d } from "../ui/3d/immediate.js";

const frame_render_event_name = "frame_render";

const REQUIRED_WEBGPU_FEATURES = ["indirect-first-instance", "subgroups"];
const WEBGPU_DIALOG_LOGO_SRC = "engine/sprites/sundown_logo.png";

function webgpu_feature_display_name(feature) {
  switch (feature) {
    case "indirect-first-instance":
      return "Indirect first instance";
    case "shader-f16":
      return "16-bit shader floats";
    case "subgroups":
      return "Shader subgroups";
    case "timestamp-query":
      return "GPU timestamp queries";
    default:
      return feature;
  }
}

function sync_canvas_size(canvas) {
  if (!canvas) {
    return;
  }

  const width = Math.max(1, canvas.clientWidth || canvas.width || window.innerWidth || 1280);
  const height = Math.max(1, canvas.clientHeight || canvas.height || window.innerHeight || 720);
  canvas.width = width;
  canvas.height = height;
}

export class Renderer {
  canvas = null;
  adapter = null;
  device = null;
  context = null;
  canvas_format = null;
  frame_number = 0;
  aspect_ratio = 1.0;
  execution_queue = new ExecutionQueue();
  render_strategy = null;
  render_strategy_type = RenderStrategyType.Deferred;
  render_graph = null;
  post_render_callbacks = [];
  pre_render_callbacks = [];

  // Renderer features
  has_f16 = false;
  has_subgroups = false;
  cvar_unsubscribers = [];

  static renderers = [];

  /**
   * Setup the renderer
   * @param {HTMLCanvasElement} canvas - The canvas to render to
   * @param {HTMLCanvasElement} canvas_ui - The canvas to render UI to
   * @param {RenderStrategy} render_strategy - The render strategy to use
   * @param {Object} options - The options for the renderer
   */
  async setup(canvas, canvas_ui, render_strategy, options = {}) {
    this.canvas = canvas;
    this.canvas_ui = canvas_ui;

    sync_canvas_size(this.canvas);
    sync_canvas_size(this.canvas_ui);

    if (!navigator.gpu) {
      this._show_webgpu_support_dialog(
        "WebGPU is not available in this browser.",
        "Sundown needs WebGPU to render the scene. Try a current Chrome, Edge, or Firefox Nightly build with hardware acceleration enabled."
      );
      throw Error("WebGPU is not supported");
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (!this.adapter) {
      this._show_webgpu_support_dialog(
        "No compatible WebGPU adapter was found.",
        "Your browser supports WebGPU, but it could not find a GPU adapter for Sundown. Check hardware acceleration and GPU blocklist settings."
      );
      throw Error("Unable to request WebGPU adapter");
    }

    this.has_f16 = this.adapter.features.has("shader-f16") && !options.use_precision_float;
    this.has_subgroups = this.adapter.features.has("subgroups");

    let required_features = [...REQUIRED_WEBGPU_FEATURES];
    if (this.has_f16) {
      required_features.push("shader-f16");
    }
    if (__DEV__) {
      required_features.push("timestamp-query");
    }

    const missing_features = required_features.filter((feature) => !this.adapter.features.has(feature));
    if (missing_features.length > 0) {
      const missing_feature_names = missing_features.map(webgpu_feature_display_name).join(", ");
      this._show_webgpu_support_dialog(
        "This GPU is missing a required WebGPU feature.",
        `Missing: ${missing_feature_names}. Sundown cannot start until these features are available in the browser and GPU driver.`
      );

      if (missing_features.includes("subgroups")) {
        throw Error("Sundown requires WebGPU subgroups to be supported");
      }

      throw Error(`Sundown requires WebGPU feature(s): ${missing_features.join(", ")}`);
    }

    try {
      this.device = await this.adapter.requestDevice({
        requiredFeatures: required_features,
        requiredLimits: {
          maxColorAttachmentBytesPerSample: 64,
          maxStorageBuffersPerShaderStage: this.adapter.limits.maxStorageBuffersPerShaderStage,
          maxStorageBufferBindingSize: this.adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize: this.adapter.limits.maxComputeWorkgroupStorageSize,
          maxTextureArrayLayers: 2048,
          maxBufferSize: this.adapter.limits.maxBufferSize,
        },
      });
    } catch (e) {
      log(e);
      log("Falling back to default limits");
      try {
        this.device = await this.adapter.requestDevice({
          requiredFeatures: required_features,
        });
      } catch (fallback_error) {
        this._show_webgpu_support_dialog(
          "Unable to create a WebGPU device.",
          "Sundown found a compatible adapter, but the browser could not create the required GPU device."
        );
        throw fallback_error;
      }
    }

    // Use lost to handle lost devices
    this.device.lost.then((info) => {
      error(`WebGPU device was lost: ${info.message}`);
    });

    this.context = this.canvas.getContext("webgpu");
    this.canvas_format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.canvas_format,
      alphaMode: "premultiplied",
    });

    await ShaderArchive.load();

    if (this.canvas_ui) {
      this.context_ui = this.canvas_ui.getContext("2d", {
        alpha: true,
      });
    }

    this.aspect_ratio = this.canvas.width / this.canvas.height;

    if (options.pointer_lock) {
      this.canvas.addEventListener("click", async () => {
        await this.canvas.requestPointerLock();
      });
    }

    this.render_graph = RenderGraph.create(this.max_bind_groups());

    this._bind_cvars();
    this._initialize_render_strategy(render_strategy);

    if (__DEV__) {
      GPUTimeQuery.init(this.device);
    }

    this._setup_resize_observer();

    Mesh.precrete_engine_primitives();
  }

  /**
   * Render the scene
   * @param {number} delta_time - The time since the last frame
   */
  render(delta_time) {
    profile_scope(frame_render_event_name, () => {
      this.advance_frame();

      this.render_graph.begin();

      this.render_strategy.draw(this.render_graph);

      flush_ui(this.context_ui);

      reset_ui_3d();
    });
  }

  /**
   * Add a callback to be called before the render graph is executed
   * @param {Function} callback - The callback to add
   */
  on_pre_render(callback) {
    this.render_graph.on_pre_render(callback);
  }

  /**
   * Enqueue a render graph command to be called before all other passes during any given render graph execution
   * @param {string} name - The name of the command
   * @param {Function} commands_callback - The callback to enqueue
   * @param {boolean} persistent - If true, the command will be called every frame
   */
  enqueue_pre_commands(name, commands_callback, persistent = false) {
    this.render_graph.queue_pre_commands(name, commands_callback, persistent);
  }

  /**
   * Unqueue a pre render graph command
   * @param {string} name - The name of the command
   */
  unqueue_pre_commands(name) {
    this.render_graph.unqueue_pre_commands(name);
  }

  /**
   * Enqueue a render graph command to be called after all other passes during any given render graph execution
   * @param {string} name - The name of the command
   * @param {Function} commands_callback - The callback to enqueue
   * @param {boolean} persistent - If true, the command will be called every frame
   */
  enqueue_post_commands(name, commands_callback, persistent = false) {
    this.render_graph.queue_post_commands(name, commands_callback, persistent);
  }

  /**
   * Unqueue a post render graph command
   * @param {string} name - The name of the command
   */
  unqueue_post_commands(name) {
    this.render_graph.unqueue_post_commands(name);
  }

  /**
   * Add a callback to be called after GPU work is completed for this frame
   * @param {Function} callback - The callback to add
   */
  on_post_render(callback) {
    this.render_graph.on_post_render(callback);
  }

  /**
   * Remove a callback from being called before the render graph is executed
   * @param {Function} callback - The callback to remove
   */
  remove_pre_render(callback) {
    this.render_graph.remove_pre_render(callback);
  }

  /**
   * Remove a callback from being called after GPU work is completed for this frame
   * @param {Function} callback - The callback to remove
   */
  remove_post_render(callback) {
    this.render_graph.remove_post_render(callback);
  }

  /**
   * Mark the bind groups for the passes as dirty
   * @param {boolean} passes_only - If true, only mark the bind groups for the passes as dirty, otherwise mark all bind groups as dirty
   */
  mark_bind_groups_dirty(passes_only = false) {
    this.render_graph.mark_pass_cache_bind_groups_dirty(passes_only);
  }

  /**
   * Force recreates all pipeline states in the render graph
   */
  recreate_pipeline_states() {
    this.render_graph.recreate_pipeline_states();
  }

  /**
   * Force recreates all resources in the render graph
   */
  refresh_render_graph(reinit = false) {
    this.render_strategy.refresh(this.render_graph, reinit);
  }

  /**
   * Only refresh the global shader bindings
   */
  refresh_global_shader_bindings() {
    const global_bindings = [
      {
        buffer: MeshData.vertex_buffer,
        offset: 0,
        size: MeshData.vertex_data ? MeshData.vertex_data.byteLength : 0,
      },
      {
        buffer: SharedViewBuffer.buffer,
        offset: 0,
        size: SharedViewBuffer.buffer_size,
      },
      {
        sampler: Texture.get_default_sampler(),
      },
      {
        sampler: TextureSampler.create({
          name: "non_filtering_sampler",
          address_mode_u: "clamp-to-edge",
          address_mode_v: "clamp-to-edge",
          address_mode_w: "clamp-to-edge",
          mag_filter: "nearest",
          min_filter: "nearest",
          mipmap_filter: "nearest",
          type: "non-filtering",
        }),
      },
      {
        sampler: TextureSampler.create({
          name: "clamped_sampler",
          address_mode_u: "clamp-to-edge",
          address_mode_v: "clamp-to-edge",
          address_mode_w: "clamp-to-edge",
          mag_filter: "linear",
          min_filter: "linear",
          mipmap_filter: "linear",
        }),
      },
      {
        sampler: TextureSampler.create({
          name: "comparison_sampler",
          mag_filter: "nearest",
          min_filter: "nearest",
          mipmap_filter: "nearest",
          compare: "less-equal",
          type: "comparison",
        }),
      },
      {
        buffer: SharedFrameInfoBuffer.buffer,
        offset: 0,
        size: SharedFrameInfoBuffer.size,
      },
    ];

    this.render_graph.queue_global_bind_group_write(global_bindings, true /* overwrite */);
  }

  /**
   * Advance the frame number
   */
  advance_frame() {
    this.frame_number++;
  }

  /**
   * Get the current frame number
   * @returns {number} - The current frame number
   */
  get_frame_number() {
    return this.frame_number;
  }

  /**
   * Get the buffered frame number
   * @returns {number} - The buffered frame number
   */
  get_buffered_frame_number() {
    return this.frame_number % MAX_BUFFERED_FRAMES;
  }

  /**
   * Get the canvas resolution
   * @returns {Object} - The canvas resolution
   */
  get_canvas_resolution() {
    return {
      width: this.canvas.width,
      height: this.canvas.height,
    };
  }

  /**
   * Check if shadows are enabled
   * @returns {boolean} - True if shadows are enabled, false otherwise
   */
  is_shadows_enabled() {
    return CVarSystem.get(EngineCVars.Renderer.ShadowsEnabled, true);
  }

  /**
   * Set the shadows enabled state
   * @param {boolean} enabled - True if shadows should be enabled, false otherwise
   */
  set_shadows_enabled(enabled) {
    CVarSystem.set(EngineCVars.Renderer.ShadowsEnabled, enabled);
  }

  /**
   * Check if global illumination is enabled
   * @returns {boolean} - True if global illumination is enabled, false otherwise
   */
  is_gi_enabled() {
    return CVarSystem.get(EngineCVars.Renderer.GIEnabled, true);
  }

  /**
   * Set the global illumination enabled state
   * @param {boolean} enabled - True if global illumination should be enabled, false otherwise
   */
  set_gi_enabled(enabled) {
    CVarSystem.set(EngineCVars.Renderer.GIEnabled, enabled);
  }

  /**
   * Get the current GI strategy type
   * @returns {GIStrategyType} - The GI strategy type
   */
  get_gi_strategy_type() {
    return CVarSystem.get(EngineCVars.Renderer.GIStrategy, GIStrategyType.DDGI);
  }

  /**
   * Set the GI strategy type
   * @param {GIStrategyType} strategy_type - The GI strategy type
   */
  set_gi_strategy_type(strategy_type) {
    CVarSystem.set(EngineCVars.Renderer.GIStrategy, strategy_type);
  }

  /**
   * Get the current AO strategy type
   * @returns {AOStrategyType} - The AO strategy type
   */
  get_ao_strategy_type() {
    return CVarSystem.get(EngineCVars.Renderer.AOStrategy, AOStrategyType.VBAO);
  }

  /**
   * Set the AO strategy type
   * @param {AOStrategyType} strategy_type - The AO strategy type
   */
  set_ao_strategy_type(strategy_type) {
    CVarSystem.set(EngineCVars.Renderer.AOStrategy, strategy_type);
  }


  /**
   * Check if reflections are enabled
   * @returns {boolean} - True if reflections are enabled, false otherwise
   */
  is_reflection_enabled() {
    return CVarSystem.get(EngineCVars.Renderer.ReflectionsEnabled, true);
  }

  /**
   * Set the reflections enabled state
   * @param {boolean} enabled - True if reflections should be enabled, false otherwise
   */
  set_reflection_enabled(enabled) {
    CVarSystem.set(EngineCVars.Renderer.ReflectionsEnabled, enabled);
  }

  /**
   * Get the current Reflection strategy type
   * @returns {ReflectionStrategyType} - The Reflection strategy type
   */
  get_reflection_strategy_type() {
    return CVarSystem.get(EngineCVars.Renderer.ReflectionStrategy, ReflectionStrategyType.SSR);
  }

  /**
   * Set the Reflection strategy type
   * @param {ReflectionStrategyType} strategy_type - The Reflection strategy type
   */
  set_reflection_strategy_type(strategy_type) {
    CVarSystem.set(EngineCVars.Renderer.ReflectionStrategy, strategy_type);
  }

  /**
   * Check if temporal anti-aliasing is enabled
   * @returns {boolean} - True if TAA is enabled, false otherwise
   */
  is_taa_enabled() {
    return CVarSystem.get(EngineCVars.Renderer.TAAEnabled, true);
  }

  /**
   * Set the TAA enabled state
   * @param {boolean} enabled - True if TAA should be enabled, false otherwise
   */
  set_taa_enabled(enabled) {
    CVarSystem.set(EngineCVars.Renderer.TAAEnabled, enabled);
  }

  /**
   * Get the current TAA feedback amount.
   * @returns {number} - Current-frame blend amount.
   */
  get_taa_feedback() {
    return CVarSystem.get(EngineCVars.Renderer.TAAFeedback, 0.5);
  }

  /**
   * Check if AO is enabled
   * @returns {boolean} - True if AO is enabled, false otherwise
   */
  is_ao_enabled() {
    return CVarSystem.get(EngineCVars.Renderer.AOEnabled, true);
  }

  /**
   * Set the AO enabled state
   * @param {boolean} enabled - True if AO should be enabled, false otherwise
   */
  set_ao_enabled(enabled) {
    CVarSystem.set(EngineCVars.Renderer.AOEnabled, enabled);
  }

  /**
   * Check if the depth prepass is enabled
   * @returns {boolean} - True if the depth prepass is enabled, false otherwise
   */
  is_depth_prepass_enabled() {
    return CVarSystem.get(EngineCVars.Renderer.DepthPrepassEnabled, true);
  }

  /**
   * Set the depth prepass enabled state
   * @param {boolean} enabled - True if the depth prepass should be enabled, false otherwise
   */
  set_depth_prepass_enabled(enabled) {
    CVarSystem.set(EngineCVars.Renderer.DepthPrepassEnabled, enabled);
  }

  /**
   * Draw a render pass
   * @param {RenderPass} render_pass - The render pass to draw
   * @param {number} triangles - The number of triangles to draw
   * @param {number} instance_count - The number of instances to draw
   */
  draw_pass(render_pass, triangles, instance_count = 1) {
    render_pass.pass.draw(triangles, instance_count);
  }

  /**
   * Set the scene ID
   * @param {number} scene_id - The scene ID
   */
  set_scene_id(scene_id) {
    this.render_graph.set_scene_id(scene_id);
  }

  /**
   * Get the debug draw type
   * @returns {DebugDrawType} - The debug draw type
   */
  get_debug_draw_type() {
    return CVarSystem.get(EngineCVars.Renderer.DebugDraw, DebugDrawType.None);
  }

  /**
   * Set the debug draw type
   * @param {DebugDrawType} debug_draw_type - The debug draw type
   */
  set_debug_draw_type(debug_draw_type) {
    CVarSystem.set(EngineCVars.Renderer.DebugDraw, debug_draw_type);
  }

  /**
   * Get the debug texture mip level (e.g. for HZB or prev lighting pyramid overlay).
   * @returns {number} - The mip level to display (0 = full resolution).
   */
  get_debug_texture_level() {
    return CVarSystem.get(EngineCVars.Renderer.DebugTextureLevel, 0);
  }

  /**
   * Set the debug texture mip level for pyramid-style debug views.
   * @param {number} level - The mip level (0 = full resolution).
   */
  set_debug_texture_level(level) {
    CVarSystem.set(EngineCVars.Renderer.DebugTextureLevel, level);
  }

  /**
   * Get the current rendering strategy type
   * @returns {RenderStrategyType} - The rendering strategy type
   */
  get_render_strategy_type() {
    return CVarSystem.get(EngineCVars.Renderer.RenderStrategy, this.render_strategy_type);
  }

  /**
   * Set the rendering strategy type
   * @param {RenderStrategyType} strategy_type - The rendering strategy type
   * @param {RenderStrategy} strategy_class - The rendering strategy class (optional, uses default for type if not provided)
   */
  set_render_strategy_type(strategy_type, strategy_class = null) {
    if (strategy_class) {
      this._apply_render_strategy_type(strategy_type, strategy_class);
      return;
    }

    CVarSystem.set(EngineCVars.Renderer.RenderStrategy, strategy_type);
  }

  /**
   * Get the current rendering strategy instance
   * @returns {RenderStrategy} - The rendering strategy instance
   */
  get_render_strategy() {
    return this.render_strategy;
  }

  /**
   * Returns namespaced scene-data handlers owned by the active strategy.
   */
  get_scene_data_handlers() {
    return this.render_strategy?.get_scene_data_handlers?.() ?? [];
  }

  /**
   * Get the maximum number of bind groups
   * @returns {number} - The maximum number of bind groups
   */
  max_bind_groups() {
    return this.adapter.limits.maxBindGroups;
  }

  /**
   * Handle the canvas being resized
   */
  on_resize() {
    sync_canvas_size(this.canvas);
    sync_canvas_size(this.canvas_ui);
    this.aspect_ratio = this.canvas.width / this.canvas.height;
  }

  /**
   * Setup the resize observer
   */
  _setup_resize_observer() {
    const observer = new ResizeObserver((entries) => {
      this.on_resize();
      this._set_shared_frame_resolution();
      global_dispatcher.dispatch("resolution_change", entries[0].contentRect);
    });
    observer.observe(this.canvas);

    this._set_shared_frame_resolution();
  }

  /**
   * Display a support dialog in case of missing support for WebGPU or features 
   */
  _show_webgpu_support_dialog(title, body) {
    const dialog_canvas = this.canvas_ui ?? this.canvas;
    if (!dialog_canvas) {
      return;
    }

    sync_canvas_size(dialog_canvas);
    const context = dialog_canvas.getContext("2d", { alpha: true });
    if (!context) {
      return;
    }

    const cached_logo = UIContext.image_cache[WEBGPU_DIALOG_LOGO_SRC];
    if (!cached_logo) {
      const logo = new Image();
      logo.onload = () => this._show_webgpu_support_dialog(title, body);
      logo.src = WEBGPU_DIALOG_LOGO_SRC;
      UIContext.image_cache[WEBGPU_DIALOG_LOGO_SRC] = logo;
    }

    context.clearRect(0, 0, dialog_canvas.width, dialog_canvas.height);
    reset_ui(dialog_canvas.width, dialog_canvas.height);

    const canvas_width = dialog_canvas.width;
    const canvas_height = dialog_canvas.height;
    const compact = canvas_width < 720 || canvas_height < 520;
    const dialog_width = Math.min(canvas_width - 32, compact ? 440 : 560);
    const content_width = Math.max(0, dialog_width - 56);
    const logo_width = Math.min(content_width, compact ? 180 : 240);
    const logo_height = Math.round(logo_width * 377 / 661);
    const dialog_padding = 28;
    const dialog_gap = compact ? 12 : 14;
    const title_font = compact ? "700 27px sans-serif" : "700 32px sans-serif";
    const body_font = compact ? "16px sans-serif" : "17px sans-serif";
    const title_line_height = compact ? 29 : 34;
    const body_line_height = compact ? 20 : 22;
    const status_height = 36;
    const measure_wrapped_height = (text, font, max_width, line_height) => {
      context.font = font;
      const words = text.split(/\s+/);
      let line_count = 1;
      let line = "";

      for (let i = 0; i < words.length; i++) {
        const test_line = line.length > 0 ? `${line} ${words[i]}` : words[i];
        if (context.measureText(test_line).width > max_width && line.length > 0) {
          line_count++;
          line = words[i];
        } else {
          line = test_line;
        }
      }

      return line_count * line_height;
    };
    const title_height = measure_wrapped_height(title, title_font, content_width, title_line_height);
    const body_height = measure_wrapped_height(body, body_font, content_width, body_line_height);
    const dialog_height =
      dialog_padding * 2 +
      logo_height +
      title_height +
      body_height +
      status_height +
      dialog_gap * 3;
    const dialog_x = Math.max(16, (canvas_width - dialog_width) / 2);
    const dialog_y = Math.max(20, (canvas_height - dialog_height) / 2);

    panel(
      {
        x: 0,
        y: 0,
        width: "100%",
        height: "100%",
        background_color: "#071117",
        z_order: 0,
      },
      () => {
        panel(
          {
            x: dialog_x,
            y: dialog_y,
            width: dialog_width,
            height: dialog_height,
            layout: "column",
            gap: dialog_gap,
            padding: dialog_padding,
            background_color: "rgba(13, 24, 31, 0.96)",
            border: "1px solid rgba(128, 226, 214, 0.38)",
            corner_radius: 8,
            box_shadow: "0 18 44 rgba(0,0,0,0.45)",
            z_order: 1,
          },
          () => {
            if (UIContext.image_cache[WEBGPU_DIALOG_LOGO_SRC]?.complete) {
              image({
                y: 0,
                src: WEBGPU_DIALOG_LOGO_SRC,
                width: logo_width,
                height: logo_height,
              });
            } else {
              label("Sundown", {
                y: 0,
                width: content_width,
                height: logo_height,
                font: compact ? "600 16px sans-serif" : "600 17px sans-serif",
                text_color: "#7ee6d7",
                text_align: "center",
                text_valign: "middle",
              });
            }
            label(title, {
              y: 0,
              width: content_width,
              height: title_height,
              font: title_font,
              text_color: "#f4fbfa",
              text_align: "left",
              text_valign: "top",
              wrap: true,
            });
            label(body, {
              y: 0,
              width: content_width,
              height: body_height,
              font: body_font,
              text_color: "rgba(227, 241, 239, 0.82)",
              text_align: "left",
              text_valign: "top",
              wrap: true,
            });
            label("The renderer stopped before the scene could start.", {
              y: 0,
              width: content_width,
              height: status_height,
              font: compact ? "600 13px monospace" : "600 14px monospace",
              text_color: "#ffd36e",
              text_align: "left",
              text_valign: "middle",
              background_color: "rgba(255, 211, 110, 0.1)",
              border: "1px solid rgba(255, 211, 110, 0.25)",
              corner_radius: 6,
              text_padding: 10,
            });
          }
        );
      }
    );

    flush_ui(context);
  }

  _bind_cvars() {
    if (this.cvar_unsubscribers.length > 0) {
      return;
    }

    const refresh_renderer = (reinit = false) => {
      if (!this.render_strategy) {
        return;
      }

      this.refresh_render_graph(reinit);
      this.recreate_pipeline_states();
    };

    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.RenderStrategy, (strategy_type) => {
        this._apply_render_strategy_type(strategy_type);
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.DepthPrepassEnabled, () => {
        refresh_renderer();
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.ShadowsEnabled, () => {
        refresh_renderer();
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.GIEnabled, () => {
        refresh_renderer();
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.AOEnabled, () => {
        refresh_renderer();
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.ReflectionsEnabled, () => {
        refresh_renderer();
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.GIStrategy, () => {
        refresh_renderer(true);
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.AOStrategy, () => {
        refresh_renderer(true);
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.ReflectionStrategy, () => {
        refresh_renderer(true);
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.TAAEnabled, (enabled) => {
        SharedViewBuffer.set_temporal_jitter_enabled(enabled);
        this.mark_bind_groups_dirty(true);
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.TAAFeedback, () => {
        this.mark_bind_groups_dirty(true);
      })
    );
    this.cvar_unsubscribers.push(
      CVarSystem.subscribe(EngineCVars.Renderer.DebugTextureLevel, () => {
        this.mark_bind_groups_dirty(true);
      })
    );
  }

  _initialize_render_strategy(strategy_class) {
    const configured_strategy = CVarSystem.get(EngineCVars.Renderer.RenderStrategy, this.render_strategy_type);
    const default_strategy = CVarSystem.get_definition(EngineCVars.Renderer.RenderStrategy)?.default_value;
    const inferred_strategy = this._infer_render_strategy_type(strategy_class);

    if (
      inferred_strategy !== null &&
      configured_strategy === default_strategy &&
      inferred_strategy !== configured_strategy
    ) {
      CVarSystem.set(EngineCVars.Renderer.RenderStrategy, inferred_strategy);
      return;
    }

    if (configured_strategy === this.render_strategy_type && strategy_class) {
      this._apply_render_strategy_type(configured_strategy, strategy_class);
      return;
    }

    this._apply_render_strategy_type(configured_strategy);
  }

  _infer_render_strategy_type(strategy_class) {
    if (strategy_class === DeferredShadingStrategy) {
      return RenderStrategyType.Deferred;
    }

    if (strategy_class === PathTracingStrategy) {
      return RenderStrategyType.PathTracing;
    }

    return null;
  }

  _apply_render_strategy_type(strategy_type, strategy_class = null) {
    if (this.render_strategy_type === strategy_type && !strategy_class) {
      return;
    }

    this.render_strategy_type = strategy_type;

    if (strategy_class) {
      this.render_strategy = new strategy_class();
      this.render_strategy.refresh(this.render_graph);
      return;
    }

    switch (strategy_type) {
      case RenderStrategyType.Deferred:
        this.render_strategy = new DeferredShadingStrategy();
        break;
      case RenderStrategyType.PathTracing:
        this.render_strategy = new PathTracingStrategy();
        break;
      default:
        log(`Unknown render strategy type: ${strategy_type}, falling back to Deferred`);
        this.render_strategy = new DeferredShadingStrategy();
        this.render_strategy_type = RenderStrategyType.Deferred;
        break;
    }

    this.render_strategy.refresh(this.render_graph);
  }

  /**
   * Set the shared frame resolution
   */
  _set_shared_frame_resolution() {
    SharedFrameInfoBuffer.set_resolution(vec2.fromValues(this.canvas.width, this.canvas.height));
  }

  /**
   * Get the renderer
   * @param {number} index - The index of the renderer
   * @returns {Renderer} - The renderer
   */
  static get(index = 0) {
    return this.renderers[index];
  }

  /**
   * Create a new renderer
   * @param {HTMLCanvasElement} canvas - The canvas to render to
   * @param {HTMLCanvasElement} canvas_ui - The canvas to render UI to
   * @param {RenderStrategy} render_strategy - The render strategy to use
   * @param {Object} options - The options for the renderer
   * @returns {number} - The index of the renderer
   */
  static async create(canvas, canvas_ui, render_strategy, options = {}) {
    const renderer = new Renderer();
    this.renderers.push(renderer);
    await renderer.setup(canvas, canvas_ui, render_strategy, options);
    return this.renderers.length - 1;
  }
}
