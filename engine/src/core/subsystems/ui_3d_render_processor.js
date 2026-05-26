import { SimulationLayer } from "../simulation_layer.js";
import { EntityFlags } from "../minimal.js";
import { EntityManager } from "../ecs/entity.js";
import { TransformFragment } from "../ecs/fragments/transform_fragment.js";
import { StaticMeshFragment } from "../ecs/fragments/static_mesh_fragment.js";
import { UserInterfaceFragment } from "../ecs/fragments/user_interface_fragment.js";
import {
  get_ui_3d_commands,
  trigger_ui_3d_event,
  UI3DCommandType,
} from "../../ui/3d/immediate.js";
import { InputProvider } from "../../input/input_provider.js";
import { InputKey } from "../../input/input_types.js";
import { RenderTaskQueue, RenderWorkKind } from "../../renderer/render_task_queue.js";
import { Mesh } from "../../renderer/mesh.js";
import { Material, MaterialTemplate } from "../../renderer/material.js";
import { TextureArrayPools } from "../../renderer/texture_pool.js";
import { ResourceCache } from "../../renderer/resource_cache.js";
import { CacheTypes, MaterialFamilyType } from "../../renderer/renderer_types.js";
import { FontCache } from "../../ui/text/font_cache.js";
import { Name } from "../../utility/names.js";
import { profile_scope } from "../../utility/performance.js";

const ui_3d_material_config = Object.freeze({
  family: MaterialFamilyType.Transparent,
  quad_template: "UI3DMaterial",
  quad_material: "UI3DMaterial",
  quad_shader: "ui_standard_material.wgsl",
  text_template: "UI3DTextMaterial",
  text_shader: "text_material.wgsl",
  image_template: "UI3DImagePoolMaterial",
  image_shader: "ui_image_material.wgsl",
  bindings: {
    ui_data: "ui_data",
    text_glyphs: "ui_text_glyphs",
    text_glyph_indices: "ui_text_glyph",
    font_glyph_data: "font_glyph_data",
    font_page_texture: "font_page_texture",
    image_texture_pool: "ui_image_texture_pool",
  },
});

const no_cull_rasterizer_state = Object.freeze({
  rasterizer_state: {
    cull_mode: "none",
  },
});

const opaque_alpha_threshold = 1.0;

/**
 * Converts an author-facing corner radius into the normalized shader value used by
 * the standard UI material. The shader reasons about a unit quad, so we normalize
 * against the smaller command extent and clamp before the rounded corners can cross.
 *
 * @param {object} command UI draw command produced by the 3D UI immediate layer.
 * @returns {number} Normalized corner radius in the range expected by the shader.
 */
function command_corner_rounding(command) {
  const min_extent = Math.max(1, Math.min(command.width ?? 1, command.height ?? 1));
  return Math.min(0.5, Math.max(0, Number(command.corner_radius ?? 0) / min_extent));
}

/**
 * Normalizes loose color-like inputs into the vec4 layout used by the UI fragment.
 * Commands may omit values freely; this keeps the fragment write path compact and
 * prevents stale per-instance color data from leaking between reused entities.
 *
 * @param {ArrayLike<number>|undefined|null} value Color-like value to normalize.
 * @param {number[]} fallback Fallback color used for omitted channels.
 * @returns {number[]} Four numeric channels suitable for fragment assignment.
 */
function vec4(value, fallback = [0, 0, 0, 0]) {
  return [
    Number(value?.[0] ?? fallback[0]),
    Number(value?.[1] ?? fallback[1]),
    Number(value?.[2] ?? fallback[2]),
    Number(value?.[3] ?? fallback[3]),
  ];
}

/**
 * Packs a vec3-style command vector into the vec4 fields used by the GPU-side UI
 * data buffer. The fourth channel acts as semantic padding: origins use `w = 1`,
 * axes use `w = 0`, mirroring common transform conventions.
 *
 * @param {ArrayLike<number>|undefined|null} value Vector-like input.
 * @param {number} w Fourth component to append.
 * @param {number[]} fallback Fallback xyz values.
 * @returns {number[]} Four numeric components for fragment assignment.
 */
function vec3_as_vec4(value, w = 0, fallback = [0, 0, 0]) {
  return [
    Number(value?.[0] ?? fallback[0]),
    Number(value?.[1] ?? fallback[1]),
    Number(value?.[2] ?? fallback[2]),
    w,
  ];
}

/**
 * Detects commands that have opted into the extensible material path. Custom
 * materials are batched conservatively by default because two commands can share a
 * shader while still requiring different textures, buffers, or uniforms.
 *
 * @param {object} command UI draw command.
 * @returns {boolean} True when the command requests a non-built-in material.
 */
function command_uses_custom_material(command) {
  return Boolean(command.material_id || command.material_template);
}

function command_parent_key(command) {
  return command.parent_entity?.id ?? "root";
}

/**
 * Picks the renderer family for built-in UI materials from authored opacity.
 *
 * Glyph coverage and rounded-corner coverage are not the same thing as material
 * opacity. Fully opaque text/panels should populate the normal lit G-buffer;
 * only commands with authored alpha below 1 need the transparent accumulator.
 *
 * @param {object} command UI draw command.
 * @returns {number} Material family for the command.
 */
function built_in_command_family(command) {
  const color = vec4(command.fill_color ?? command.color, [1, 1, 1, 1]);
  return color[3] >= opaque_alpha_threshold
    ? MaterialFamilyType.Opaque
    : MaterialFamilyType.Transparent;
}

/**
 * Small per-frame view of GPU resources that every UI material may need.
 *
 * The underlying fragment buffers can be recreated or resized by the ECS, so
 * providers should not cache raw buffer handles indefinitely. Instead the registry
 * refreshes this object once per frame and passes it into providers as the stable
 * source of truth for shared bindings.
 */
class UI3DFrameResources {
  ui_data = null;
  text_glyphs = null;

  /**
   * Refreshes shared fragment GPU buffers for the current frame.
   *
   * This is intentionally separate from material creation. Materials can live
   * across frames, but their storage bindings must track the current ECS buffers.
   *
   * @returns {void}
   */
  refresh() {
    this.ui_data = EntityManager.get_fragment_gpu_buffer(
      UserInterfaceFragment,
      ui_3d_material_config.bindings.ui_data
    )?.buffer;
    this.text_glyphs = EntityManager.get_fragment_gpu_buffer(
      UserInterfaceFragment,
      ui_3d_material_config.bindings.text_glyph_indices
    )?.buffer;
  }
}

/**
 * Base class for a family of UI materials.
 *
 * Providers keep shader/template/material knowledge out of the render processor.
 * Each provider answers two questions: "can I service this command?" and "which
 * material should render it?" This makes new UI shaders additive instead of
 * turning the processor into a growing switch statement.
 */
class UI3DMaterialProvider {
  material_ids = new Map();

  /**
   * Gives providers a chance to create templates and rebind frame-scoped buffers.
   *
   * @param {UI3DFrameResources} frame_resources Shared GPU resources for this frame.
   * @returns {void}
   */
  prepare_frame(frame_resources) { }

  /**
   * Returns whether this provider knows how to resolve a command.
   *
   * @param {object} command UI draw command.
   * @returns {boolean} True when this provider should handle the command.
   */
  supports(command) {
    return false;
  }

  /**
   * Resolves or creates the material that should render a command.
   *
   * @param {object} command UI draw command.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources for this frame.
   * @returns {number|null} Material id, or null when the command cannot render yet.
   */
  material_for_command(command, frame_resources) {
    return null;
  }

  /**
   * Creates a transparent, no-cull material template if it does not already exist.
   *
   * @param {string} name Material template name.
   * @param {string} shader Shader asset path.
   * @param {number} family Renderer material family.
   * @returns {void}
   */
  _ensure_template(name, shader, family = ui_3d_material_config.family) {
    if (!MaterialTemplate.get_template(name, family)) {
      MaterialTemplate.create(name, shader, family, no_cull_rasterizer_state);
    }
  }

  /**
   * Binds the shared per-instance UI data buffer to a material.
   *
   * Most UI shaders need this buffer because it carries quad placement, color,
   * rounded-corner parameters, and other instance data emitted by this processor.
   *
   * @param {number} material_id Material resource id.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources.
   * @returns {void}
   */
  _bind_shared_ui_data(material_id, frame_resources) {
    const material = Material.get(material_id);
    if (frame_resources.ui_data) {
      material?.set_storage_data(ui_3d_material_config.bindings.ui_data, frame_resources.ui_data);
      material?.listen_for_storage_data(ui_3d_material_config.bindings.ui_data);
    }
  }
}

/**
 * Provider for the default rounded-quad UI material.
 *
 * This is the common path for panels, rects, buttons, and other filled/bordered
 * UI primitives. A single material instance can render all standard quads because
 * their per-element differences live in the `ui_data` fragment buffer.
 */
class UI3DStandardMaterialProvider extends UI3DMaterialProvider {
  material_ids = new Map();

  /**
   * Ensures the standard template/material exist and are bound to current buffers.
   *
   * @param {UI3DFrameResources} frame_resources Shared GPU resources for this frame.
   * @returns {void}
   */
  prepare_frame(frame_resources) {
    this._ensure_material(MaterialFamilyType.Opaque);
    this._ensure_material(MaterialFamilyType.Transparent);
    for (const material_id of this.material_ids.values()) {
      this._bind_shared_ui_data(material_id, frame_resources);
    }
  }

  /**
   * Standard quads are handled here unless a custom material provider takes them first.
   *
   * @param {object} command UI draw command.
   * @returns {boolean} True for standard quad commands.
   */
  supports(command) {
    return command.type === UI3DCommandType.Quad;
  }

  /**
   * Returns the shared standard UI material.
   *
   * @param {object} command UI draw command.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources.
   * @returns {number} Standard UI material id.
   */
  material_for_command(command, frame_resources) {
    const family = built_in_command_family(command);
    const material_id = this._ensure_material(family);
    this._bind_shared_ui_data(material_id, frame_resources);
    return material_id;
  }

  /**
   * Lazily creates the standard UI material instance.
   *
   * @returns {void}
   */
  _ensure_material(family) {
    this._ensure_template(ui_3d_material_config.quad_template, ui_3d_material_config.quad_shader, family);

    let material_id = this.material_ids.get(family);
    if (!material_id) {
      material_id = Material.create(
        `${ui_3d_material_config.quad_material}_${family}`,
        ui_3d_material_config.quad_template,
        { family }
      );
      this.material_ids.set(family, material_id);
    }
    return material_id;
  }
}

/**
 * Provider for MSDF text glyph rendering.
 *
 * Text differs from regular quads because it needs a font glyph buffer and a font
 * atlas texture in addition to the normal `ui_data` buffer. Materials are cached
 * per font/page texture so each text batch can bind the right atlas while still
 * using instanced quads for individual glyphs.
 */
class UI3DTextMaterialProvider extends UI3DMaterialProvider {
  /**
   * Ensures text template availability and refreshes frame buffer bindings.
   *
   * @param {UI3DFrameResources} frame_resources Shared GPU resources for this frame.
   * @returns {void}
   */
  prepare_frame(frame_resources) {
    this._ensure_template(
      ui_3d_material_config.text_template,
      ui_3d_material_config.text_shader,
      MaterialFamilyType.Opaque
    );
    this._ensure_template(
      ui_3d_material_config.text_template,
      ui_3d_material_config.text_shader,
      MaterialFamilyType.Transparent
    );

    for (const material_id of this.material_ids.values()) {
      this._bind_text_material_buffers(material_id, frame_resources);
    }
  }

  /**
   * Claims glyph commands emitted by the 3D UI text layout path.
   *
   * @param {object} command UI draw command.
   * @returns {boolean} True for text glyph commands.
   */
  supports(command) {
    return command.type === UI3DCommandType.Text;
  }

  /**
   * Resolves the font/page material for a glyph command.
   *
   * If the font has not loaded yet, this returns null and the glyph is skipped for
   * the frame. Once the font cache is ready the same command stream will produce a
   * material without needing special retry logic.
   *
   * @param {object} command UI text glyph command.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources.
   * @returns {number|null} Text material id, or null while font data is unavailable.
   */
  material_for_command(command, frame_resources) {
    const font = FontCache.get_font_object(command.font_id);
    if (!font) {
      return null;
    }

    const texture_id = command.font_texture ? Name.from(command.font_texture) : 0;
    const family = built_in_command_family(command);
    const material_key = `${command.font_id}|${texture_id}|${family}`;
    let material_id = this.material_ids.get(material_key);
    if (!material_id) {
      material_id = this._create_text_material(command.font_id, texture_id, font, family);
      this.material_ids.set(material_key, material_id);
    }

    this._bind_text_material_buffers(material_id, frame_resources);
    return material_id;
  }

  /**
   * Creates a material bound to a specific font glyph buffer and atlas page.
   *
   * @param {number} font_id Loaded font id.
   * @param {number} texture_id Font atlas texture id.
   * @param {object} font Loaded font object from FontCache.
   * @returns {number} Created material id.
   */
  _create_text_material(font_id, texture_id, font, family) {
    const material_id = Material.create(
      `UI3DTextMaterial_${font_id}_${texture_id}_${family}`,
      ui_3d_material_config.text_template,
      { family }
    );
    const material = Material.get(material_id);

    material?.set_storage_data(
      ui_3d_material_config.bindings.font_glyph_data,
      font.font_glyph_data_buffer
    );
    material?.set_texture_data(
      ui_3d_material_config.bindings.font_page_texture,
      texture_id ? ResourceCache.get().fetch(CacheTypes.IMAGE, texture_id) : null
    );

    return material_id;
  }

  /**
   * Rebinds frame-scoped buffers used by text shaders.
   *
   * @param {number} material_id Text material id.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources.
   * @returns {void}
   */
  _bind_text_material_buffers(material_id, frame_resources) {
    const material = Material.get(material_id);
    if (frame_resources.ui_data) {
      material?.set_storage_data(ui_3d_material_config.bindings.ui_data, frame_resources.ui_data);
      material?.listen_for_storage_data(ui_3d_material_config.bindings.ui_data);
    }
    if (frame_resources.text_glyphs) {
      material?.set_storage_data(
        ui_3d_material_config.bindings.text_glyphs,
        frame_resources.text_glyphs
      );
      material?.listen_for_storage_data(
        ui_3d_material_config.bindings.text_glyph_indices,
        (_, buffer) => material.set_storage_data(ui_3d_material_config.bindings.text_glyphs, buffer)
      );
    }
  }
}

class UI3DImageMaterialProvider extends UI3DMaterialProvider {
  pool_key = "ui";

  prepare_frame(frame_resources) {
    this._ensure_template(
      ui_3d_material_config.image_template,
      ui_3d_material_config.image_shader,
      MaterialFamilyType.Opaque
    );
    this._ensure_template(
      ui_3d_material_config.image_template,
      ui_3d_material_config.image_shader,
      MaterialFamilyType.Transparent
    );

    for (const material_id of this.material_ids.values()) {
      this._bind_image_material_buffers(material_id, frame_resources);
    }
  }

  supports(command) {
    return command.type === UI3DCommandType.Image;
  }

  material_for_command(command, frame_resources) {
    if (!command.image_texture) {
      return null;
    }

    const family = built_in_command_family(command);
    const material_key = `${ui_3d_material_config.image_template}_${family}`;
    let material_id = this.material_ids.get(material_key);
    if (!material_id) {
      material_id = Material.create(
        `${ui_3d_material_config.image_template}_${family}`,
        ui_3d_material_config.image_template,
        { family }
      );
      this.material_ids.set(material_key, material_id);
    }

    this._bind_image_material_buffers(material_id, frame_resources);
    return material_id;
  }

  _bind_image_material_buffers(material_id, frame_resources) {
    const material = Material.get(material_id);
    this._bind_shared_ui_data(material_id, frame_resources);
    material?.set_texture_data(
      ui_3d_material_config.bindings.image_texture_pool,
      this._get_texture_pool()
    );
    material?.listen_for_texture_data(`texture_pool_${this.pool_key}`, () => {
      material.set_texture_data(
        ui_3d_material_config.bindings.image_texture_pool,
        this._get_texture_pool()
      );
    });
  }

  _get_texture_pool() {
    return (
      ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(`texture_pool_${this.pool_key}`)) ??
      TextureArrayPools.get_fallback_view()
    );
  }
}

/**
 * Provider for caller-supplied UI materials.
 *
 * This keeps the built-in processor open-ended: tools can pass a prebuilt
 * `material_id`, or name a `material_template` plus optional shader and bindings.
 * The provider still binds `ui_data` by default so custom shaders can reuse the
 * same placement and per-element data contract as standard UI materials.
 */
class UI3DCustomMaterialProvider extends UI3DMaterialProvider {
  /**
   * Claims commands that explicitly request a material or material template.
   *
   * @param {object} command UI draw command.
   * @returns {boolean} True for custom-material commands.
   */
  supports(command) {
    return Boolean(command.material_id || command.material_template);
  }

  /**
   * Resolves a caller-specified material and applies command-level bindings.
   *
   * @param {object} command UI draw command.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources.
   * @returns {number|null} Material id, or null if the template cannot be resolved.
   */
  material_for_command(command, frame_resources) {
    const material_id = this._resolve_material_id(command.material_id) ?? this._material_from_template(command);
    if (material_id) {
      this._bind_command_resources(material_id, command, frame_resources);
      return material_id;
    }
    return null;
  }

  /**
   * Accepts either numeric ids or material names for convenience at call sites.
   *
   * @param {number|string|null|undefined} material_id Material id or material name.
   * @returns {number|null|undefined} Numeric material id when resolvable.
   */
  _resolve_material_id(material_id) {
    return typeof material_id === "string" ? Name.from(material_id) : material_id;
  }

  /**
   * Creates or reuses a material from a caller-specified template.
   *
   * If `material_shader` is provided, the template is created on demand. Without a
   * shader, the template must already exist so we do not guess at shader assets.
   *
   * @param {object} command UI draw command.
   * @returns {number|null} Material id, or null if no usable template exists.
   */
  _material_from_template(command) {
    if (!command.material_template) {
      return null;
    }

    const family = command.material_family ?? ui_3d_material_config.family;
    if (command.material_shader) {
      this._ensure_template(command.material_template, command.material_shader, family);
    } else if (!MaterialTemplate.get_template(command.material_template, family)) {
      return null;
    }

    const material_key =
      command.material_key ??
      command.material_name ??
      `${command.material_template}_${command.type}`;
    let material_id = this.material_ids.get(material_key);
    if (!material_id) {
      material_id = Material.create(material_key, command.material_template, { family });
      this.material_ids.set(material_key, material_id);
    }
    return material_id;
  }

  /**
   * Applies shared UI data plus custom storage/texture/uniform maps.
   *
   * @param {number} material_id Material id to bind.
   * @param {object} command UI draw command.
   * @param {UI3DFrameResources} frame_resources Shared GPU resources.
   * @returns {void}
   */
  _bind_command_resources(material_id, command, frame_resources) {
    const material = Material.get(material_id);
    this._bind_shared_ui_data(material_id, frame_resources);
    this._bind_storage_map(material, command.storage_bindings ?? command.buffers);
    this._bind_texture_map(material, command.texture_bindings ?? command.textures);
    this._bind_uniform_map(material, command.uniform_bindings ?? command.uniforms);
  }

  /**
   * Binds arbitrary storage buffers by shader binding name.
   *
   * @param {Material|null} material Material instance.
   * @param {Object<string, *>} bindings Storage binding map.
   * @returns {void}
   */
  _bind_storage_map(material, bindings) {
    if (!material || !bindings) {
      return;
    }

    for (const [name, data] of Object.entries(bindings)) {
      material.set_storage_data(name, data);
    }
  }

  /**
   * Binds arbitrary textures by shader binding name.
   *
   * @param {Material|null} material Material instance.
   * @param {Object<string, *>} bindings Texture binding map.
   * @returns {void}
   */
  _bind_texture_map(material, bindings) {
    if (!material || !bindings) {
      return;
    }

    for (const [name, texture] of Object.entries(bindings)) {
      material.set_texture_data(name, this._resolve_texture(texture));
    }
  }

  /**
   * Binds arbitrary uniform payloads by shader binding name.
   *
   * @param {Material|null} material Material instance.
   * @param {Object<string, *>} bindings Uniform binding map.
   * @returns {void}
   */
  _bind_uniform_map(material, bindings) {
    if (!material || !bindings) {
      return;
    }

    for (const [name, data] of Object.entries(bindings)) {
      material.set_uniform_data(name, data);
    }
  }

  /**
   * Resolves texture names/ids through the resource cache while passing through
   * already-loaded texture objects unchanged.
   *
   * @param {string|number|object|null} texture Texture name, id, or texture object.
   * @returns {object|null|undefined} Texture resource for material binding.
   */
  _resolve_texture(texture) {
    if (typeof texture === "string") {
      return ResourceCache.get().fetch(CacheTypes.IMAGE, Name.from(texture));
    }
    if (typeof texture === "number") {
      return ResourceCache.get().fetch(CacheTypes.IMAGE, texture);
    }
    return texture;
  }
}

/**
 * Ordered provider registry for UI material resolution.
 *
 * The registry is intentionally ordered. Custom providers run first so explicit
 * caller intent wins, then text, then the standard quad fallback. External systems
 * can register providers at the front to add specialized material families without
 * editing this processor.
 */
class UI3DMaterialRegistry {
  frame_resources = new UI3DFrameResources();
  providers = [
    new UI3DCustomMaterialProvider(),
    new UI3DImageMaterialProvider(),
    new UI3DTextMaterialProvider(),
    new UI3DStandardMaterialProvider(),
  ];

  /**
   * Refreshes shared buffers and lets each provider perform frame setup.
   *
   * @returns {void}
   */
  prepare_frame() {
    this.frame_resources.refresh();
    for (const provider of this.providers) {
      provider.prepare_frame(this.frame_resources);
    }
  }

  /**
   * Finds the first provider that supports a command and asks it for a material.
   *
   * @param {object} command UI draw command.
   * @returns {number|null} Material id, or null when no provider can render it.
   */
  material_for_command(command) {
    const provider = this.providers.find((candidate) => candidate.supports(command));
    return provider?.material_for_command(command, this.frame_resources) ?? null;
  }

  /**
   * Adds a higher-priority provider to the front of the registry.
   *
   * @param {UI3DMaterialProvider} provider Provider-like object.
   * @returns {void}
   */
  register_provider(provider) {
    this.providers.unshift(provider);
  }
}

/**
 * Simulation layer that turns 3D UI commands into instanced render work.
 *
 * The immediate UI layer owns layout and command emission. This processor owns the
 * render-facing half: material selection, batching compatible commands into ECS
 * instances, writing fragment data, and submitting mesh tasks. Keeping those roles
 * separate lets UI code stay ergonomic while the renderer receives compact batches.
 */
export class UI3DRenderProcessor extends SimulationLayer {
  entities = [];
  entity_query = null;
  material_registry = new UI3DMaterialRegistry();
  scene = null;

  /**
   * Builds the event query and binds profiled update implementations once.
   *
   * The query covers both persistent 3D UI entities and the reusable
   * instanced entities generated by this processor. Immediate-rendered instances
   * generally do not opt into cursor events, but sharing one query keeps all 3D UI
   * interaction state in the same subsystem.
   *
   * @returns {void}
   */
  init() {
    this.entity_query = EntityManager.create_query([UserInterfaceFragment]);
    this._update_internal = this._update_internal.bind(this);
    this._process_cursor_events = this._process_cursor_events.bind(this);
  }

  /**
   * Runs 3D UI render preparation and interaction processing for the frame.
   *
   * @param {number} delta_time Simulation delta time, currently unused here.
   * @returns {void}
   */
  post_update(delta_time) {
    super.post_update(delta_time);
    profile_scope("UI3DRenderProcessor.update", this._update_internal);
    profile_scope("UI3DRenderProcessor.cursor_events", this._process_cursor_events);
  }

  /**
   * Stores the scene used for cursor picking lookups.
   *
   * Persistent 3D UI event processing compares each UI entity against the
   * entity currently under the cursor. Keeping this on the centralized 3D UI layer
   * avoids a second subsystem whose only job is to mirror UI fragment state.
   *
   * @param {object} scene Active scene instance.
   * @returns {void}
   */
  set_scene(scene) {
    this.scene = scene;
  }

  /**
   * Allows feature code to add a specialized material resolver.
   *
   * Registered providers are checked before built-ins, which makes this suitable
   * for domain-specific UI shaders that should override standard behavior.
   *
   * @param {UI3DMaterialProvider} provider Provider-like material resolver.
   * @returns {void}
   */
  register_material_provider(provider) {
    this.material_registry.register_provider(provider);
  }

  /**
   * Converts the current command buffer into render queue tasks.
   *
   * This method first clears stale UI work, refreshes material resources, batches
   * compatible commands, then writes each batch into one instanced ECS entity.
   *
   * @returns {void}
   */
  _update_internal() {
    const commands = get_ui_3d_commands();

    RenderTaskQueue.remove_tasks(RenderWorkKind.UI3DMesh);

    this.material_registry.prepare_frame();

    const mesh = Mesh.quad();
    const mesh_id = Name.from(mesh.name);
    const batches = this._build_command_batches(commands);

    let entity_index = 0;
    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      const entity = this._get_entity(entity_index++, batch.commands.length, batch.parent_entity);
      this._write_command_batch(batch.commands, entity);
      RenderTaskQueue.submit({
        lane_id: RenderWorkKind.UI3DMesh,
        mesh_id,
        entity,
        material_id: batch.material_id,
        section: 0,
      });
    }
  }

  /**
   * Updates cursor state and dispatches 3D UI events.
   *
   * This remains separate from the render batching methods because it operates
   * over existing ECS UI fragments, not over the transient command buffer emitted
   * by `immediate.js`.
   *
   * @returns {void}
   */
  _process_cursor_events() {
    if (!this.scene?.get_cursor_pixel_entity) {
      return;
    }

    const cursor_entity = this.scene.get_cursor_pixel_entity();
    this.entity_query.for_each((chunk, slot, instance_count) => {
      const entity_flags = chunk.flags_meta[slot];
      if ((entity_flags & EntityFlags.ALIVE) === 0) {
        return;
      }

      const user_interfaces = chunk.get_fragment_view(UserInterfaceFragment);
      const entity = EntityManager.get_entity_for(chunk, slot);

      for (let i = 0; i < instance_count; ++i) {
        const index = slot + i;
        if (!user_interfaces.allows_cursor_events[index]) {
          continue;
        }

        user_interfaces.was_cursor_inside[index] = user_interfaces.is_cursor_inside[index];
        user_interfaces.is_cursor_inside[index] = entity === cursor_entity;

        user_interfaces.was_clicked[index] = user_interfaces.is_clicked[index];
        user_interfaces.is_clicked[index] =
          user_interfaces.is_cursor_inside[index] &&
          InputProvider.get_action(InputKey.B_mouse_left);

        user_interfaces.was_pressed[index] = user_interfaces.is_pressed[index];
        user_interfaces.is_pressed[index] =
          user_interfaces.is_cursor_inside[index] &&
          InputProvider.get_state(InputKey.B_mouse_left);

        this._dispatch_cursor_events(entity, user_interfaces, index);
        this._mark_cursor_event_fields_dirty(chunk);
      }
    });
  }

  /**
   * Emits high-level 3D UI callbacks for cursor transitions and button state.
   *
   * @param {object} entity Entity whose UI fragment is being updated.
   * @param {object} user_interfaces UserInterface fragment view for the chunk.
   * @param {number} index Absolute row index inside the chunk.
   * @returns {void}
   */
  _dispatch_cursor_events(entity, user_interfaces, index) {
    if (!user_interfaces.was_cursor_inside[index] && user_interfaces.is_cursor_inside[index]) {
      trigger_ui_3d_event(entity, "hover");
    } else if (
      user_interfaces.was_cursor_inside[index] &&
      !user_interfaces.is_cursor_inside[index]
    ) {
      trigger_ui_3d_event(entity, "leave");
    }

    if (user_interfaces.is_clicked[index]) {
      trigger_ui_3d_event(entity, "selected");
      this._consume_cursor_event_if_requested(user_interfaces, index);
    }
    if (user_interfaces.is_pressed[index]) {
      trigger_ui_3d_event(entity, "pressed");
      this._consume_cursor_event_if_requested(user_interfaces, index);
    }
  }

  /**
   * Consumes mouse input for UI elements that request event capture.
   *
   * @param {object} user_interfaces UserInterface fragment view.
   * @param {number} index Absolute row index inside the chunk.
   * @returns {void}
   */
  _consume_cursor_event_if_requested(user_interfaces, index) {
    if (user_interfaces.consume_events[index]) {
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

  /**
   * Marks CPU-side cursor/event fields dirty after mutating a UI fragment row.
   *
   * @param {object} chunk ECS chunk containing the updated row.
   * @returns {void}
   */
  _mark_cursor_event_fields_dirty(chunk) {
    chunk.mark_dirty("was_cursor_inside");
    chunk.mark_dirty("is_cursor_inside");
    chunk.mark_dirty("was_clicked");
    chunk.mark_dirty("is_clicked");
    chunk.mark_dirty("was_pressed");
    chunk.mark_dirty("is_pressed");
  }

  /**
   * Groups commands that can share one instanced entity and render task.
   *
   * The batch key combines material id with an optional command-level key. Text
   * supplies a key per string, standard quads share by default, and custom material
   * commands default to one-command batches unless callers opt into sharing.
   *
   * @param {object[]} commands Commands emitted by the 3D UI immediate layer.
   * @returns {{material_id:number, commands:object[]}[]} Ordered command batches.
   */
  _build_command_batches(commands) {
    const batches = [];
    const batch_by_key = new Map();

    for (let i = 0; i < commands.length; i++) {
      const command = commands[i];
      if (!command) continue;

      const material_id = this.material_registry.material_for_command(command);
      if (!material_id) continue;

      const batch_key = this._command_batch_key(command, material_id);
      let batch = batch_by_key.get(batch_key);
      if (!batch) {
        batch = { material_id, parent_entity: command.parent_entity ?? null, commands: [] };
        batch_by_key.set(batch_key, batch);
        batches.push(batch);
      }
      batch.commands.push(command);
    }

    return batches;
  }

  /**
   * Produces the logical grouping key for command instancing.
   *
   * @param {object} command UI draw command.
   * @param {number} material_id Resolved material id for the command.
   * @returns {string} Stable key used within the current frame.
   */
  _command_batch_key(command, material_id) {
    const local_key =
      command.batch_key ??
      (command_uses_custom_material(command) ? `command_${command.order}` : "shared");
    return `${material_id}|${command_parent_key(command)}|${local_key}`;
  }

  /**
   * Returns a reusable ECS entity sized for a batch.
   *
   * Entities persist across frames so UI rendering avoids churn. When a batch grows
   * or shrinks, the ECS instance count is adjusted and fragment rows are reused.
   *
   * @param {number} index Reuse-pool index for this frame's batch.
   * @param {number} instance_count Number of command instances to store.
   * @returns {object} ECS entity handle.
   */
  _get_entity(index, instance_count, parent_entity = null) {
    while (this.entities.length <= index) {
      const entity = EntityManager.create_entity(
        [TransformFragment, StaticMeshFragment, UserInterfaceFragment],
        instance_count
      );
      EntityManager.set_entity_parent(entity, parent_entity);
      EntityManager.set_entity_flags(
        entity,
        EntityFlags.ALIVE |
        EntityFlags.HAS_MESH |
        EntityFlags.IGNORE_PARENT_SCALE |
        EntityFlags.IGNORE_TLAS
      );
      this.entities.push(entity);
    }

    const entity = this.entities[index];
    if (EntityManager.get_entity_parent(entity) !== parent_entity) {
      EntityManager.set_entity_parent(entity, parent_entity);
    }
    if (entity.instance_count !== instance_count) {
      EntityManager.set_entity_instance_count(entity, instance_count);
    }
    return entity;
  }

  /**
   * Writes every command in a batch into a matching ECS instance slot.
   *
   * @param {object[]} commands Commands sharing one material/batch key.
   * @param {object} entity Instanced ECS entity that backs the batch.
   * @returns {void}
   */
  _write_command_batch(commands, entity) {
    for (let i = 0; i < commands.length; i++) {
      this._update_entity_transform(commands[i], entity, i);
      this._write_command_fragment(commands[i], entity, i);
    }
  }

  /**
   * Updates the transform fragment for one UI quad instance.
   *
   * The shader uses `ui_origin`, `ui_x_axis`, and `ui_y_axis` for exact placement;
   * the transform still carries a conservative center/scale so existing renderer
   * systems have a conventional object transform to work with.
   *
   * @param {object} command UI draw command.
   * @param {object} entity Instanced ECS entity.
   * @param {number} instance Instance index inside the entity.
   * @returns {void}
   */
  _update_entity_transform(command, entity, instance = 0) {
    const transform = EntityManager.get_fragment(entity, TransformFragment, instance);
    if (!transform) {
      return;
    }

    const origin = command.origin ?? [0, 0, 0];
    const x_axis = command.x_axis ?? [1, 0, 0];
    const y_axis = command.y_axis ?? [0, 1, 0];
    const center = [
      Number(origin[0] ?? 0) + Number(x_axis[0] ?? 0) * 0.5 + Number(y_axis[0] ?? 0) * 0.5,
      Number(origin[1] ?? 0) + Number(x_axis[1] ?? 0) * 0.5 + Number(y_axis[1] ?? 0) * 0.5,
      Number(origin[2] ?? 0) + Number(x_axis[2] ?? 0) * 0.5 + Number(y_axis[2] ?? 0) * 0.5,
      1,
    ];
    const x_len = Math.hypot(
      Number(x_axis[0] ?? 0),
      Number(x_axis[1] ?? 0),
      Number(x_axis[2] ?? 0)
    );
    const y_len = Math.hypot(
      Number(y_axis[0] ?? 0),
      Number(y_axis[1] ?? 0),
      Number(y_axis[2] ?? 0)
    );
    command._ui_center = center;
    command._ui_radius_scale = Math.max(0.001, x_len * 0.5, y_len * 0.5);

    transform.position = center;
    transform.rotation = [0, 0, 0, 1];
    transform.scale = [
      command._ui_radius_scale,
      command._ui_radius_scale,
      command._ui_radius_scale,
      1,
    ];
  }

  /**
   * Writes shader-facing UI data for one command instance.
   *
   * These fragment fields are consumed by the standard and text UI shaders, and
   * are also available to custom materials through the shared `ui_data` binding.
   *
   * @param {object} command UI draw command.
   * @param {object} entity Instanced ECS entity.
   * @param {number} instance Instance index inside the entity.
   * @returns {void}
   */
  _write_command_fragment(command, entity, instance = 0) {
    const ui = EntityManager.get_fragment(entity, UserInterfaceFragment, instance);
    if (!ui) {
      return;
    }

    const color = vec4(command.fill_color ?? command.color, [1, 1, 1, 1]);
    const emissive = Number(command.emissive ?? 0);
    const center = command._ui_center ?? [0, 0, 0];
    const radius_scale = command._ui_radius_scale ?? 1;
    const origin = command.origin ?? [0, 0, 0];
    const x_axis = command.x_axis ?? [1, 0, 0];
    const y_axis = command.y_axis ?? [0, 1, 0];
    ui.ui_origin = [
      (Number(origin[0] ?? 0) - Number(center[0] ?? 0)) / radius_scale,
      (Number(origin[1] ?? 0) - Number(center[1] ?? 0)) / radius_scale,
      (Number(origin[2] ?? 0) - Number(center[2] ?? 0)) / radius_scale,
      1,
    ];
    ui.ui_x_axis = [
      Number(x_axis[0] ?? 0) / radius_scale,
      Number(x_axis[1] ?? 0) / radius_scale,
      Number(x_axis[2] ?? 0) / radius_scale,
      0,
    ];
    ui.ui_y_axis = [
      Number(y_axis[0] ?? 0) / radius_scale,
      Number(y_axis[1] ?? 0) / radius_scale,
      Number(y_axis[2] ?? 0) / radius_scale,
      0,
    ];
    ui.ui_uv_rect = [
      0,
      0,
      Number(command.page_texture_size?.[0] ?? 1),
      Number(command.page_texture_size?.[1] ?? 1),
    ];
    ui.ui_color = color;
    ui.ui_border_color = vec4(command.border_color, [0, 0, 0, 0]);
    ui.ui_params = [
      command_corner_rounding(command),
      Number(command.border_width ?? 0),
      emissive,
      command.type === UI3DCommandType.Image
        ? Math.max(0, Number(command.image_texture?.bindless_handle ?? 0))
        : Math.max(
          0.0001,
          Number(command.width ?? 1) / Math.max(0.0001, Number(command.height ?? 1))
        ),
    ];
    ui.ui_text_glyph = Number(command.glyph_index ?? 0);

    ui.ui_emissive = emissive;
    ui.ui_rounding = command_corner_rounding(command);
    
    ui.chunk.mark_dirty(ui_3d_material_config.bindings.ui_data);
  }
}
