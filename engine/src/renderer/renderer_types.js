const graphics_pass_name = "Graphics";
const present_pass_name = "Present";
const compute_pass_name = "Compute";
const graph_local_pass_name = "GraphLocal";

/**
 * Flags for image resources in the render graph.
 * @enum {number}
 */
export const ImageFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a transient image resource */
  Transient: 1,
  /** Indicates the image is loaded locally */
  LocalLoad: 2,
});

/**
 * Types of shader resources.
 * @enum {number}
 */
export const ShaderResourceType = Object.freeze({
  Uniform: 0,
  Storage: 1,
  Texture: 2,
  Sampler: 3,
  StorageTexture: 4,
});

/**
 * Types of resources in the resource cache.
 * @enum {number}
 */
export const CacheTypes = Object.freeze({
    SHADER: 0,
    PIPELINE_STATE: 1,
    RENDER_PASS: 2,
    BIND_GROUP: 3,
    BIND_GROUP_LAYOUT: 4,
    BUFFER: 5,
    IMAGE: 6,
    IMAGE_POOL: 7,
    SAMPLER: 8,
    MESH: 9,
    MATERIAL: 10,
});

/**
 * Flags for render passes in the render graph.
 * @enum {number}
 */
export const RenderPassFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a graphics pass */
  Graphics: 1,
  /** Indicates a present pass */
  Present: 2,
  /** Indicates a compute pass */
  Compute: 4,
  /** Indicates a graph-local pass */
  GraphLocal: 8,
});

/**
 * Types of material families.
 * @enum {number}
 */
export const MaterialFamilyType = Object.freeze({
  Opaque: 0,
  Transparent: 1,
});

/**
 * Flags for buffer resources in the render graph.
 * @enum {number}
 */
export const BufferFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a transient buffer resource */
  Transient: 1,
  /** Indicates a buffer that needs to refresh global shader bindings */
  GlobalBinding: 2,
});

/**
 * Index of bindless group for image resources.
 * @enum {number}
 */
export const BindlessGroupIndex = Object.freeze({
    Image: 0,
    StorageImage: 1
}); 

/**
 * Types of bind groups in the render graph.
 * @enum {number}
 */
export const BindGroupType = Object.freeze({
    Global: 0,
    Pass: 1,
    Material: 2,
    Num: 3
});

/**
 * Types of rendering strategies.
 * @enum {number}
 */
export const RenderStrategyType = Object.freeze({
    Deferred: 0,
    PathTracing: 1,
});

/**
 * Types of GI strategies.
 * @enum {number}
 */
export const GIStrategyType = Object.freeze({
    PTGI: 0,
    DDGI: 1,
});

/**
 * Types of AO strategies.
 * @enum {number}
 */
export const AOStrategyType = Object.freeze({
    GTAO: 0,
    RTAO: 1,
});

/**
 * Types of Reflection strategies.
 * @enum {number}
 */
export const ReflectionStrategyType = Object.freeze({
    SSR: 0,
});

/**
 * Types of debug draw in the render graph.
 * @enum {number}
 */
export const DebugDrawType = Object.freeze({
    None: 0,
    Wireframe: 1,
    Depth: 2,
    Normal: 3,
    Emissive: 4,
    Motion: 5,
    EntityId: 6,
    HZB: 7,
    ASVSM_ShadowAtlas: 8,
    ASVSM_ShadowPageTable: 9,
    ASVSM_TileOverlay: 10,
    ASVSM_TileRenderOutput: 11,
    ASVSM_DirtyTiles: 12,
    Bloom: 13,
    AO: 14,
    BentNormal: 15,
    GI_Direct: 16,
    GI_Diffuse: 17,
    GI_Specular: 18,
    GI_WorldCache: 19,
    GI_Probes: 20,
    GI_Reflections: 21,
    EntityBounds: 22,
    BVH: 23,
    BLAS_Bounds: 25,
});

/**
 * Channels for texture sampling.
 * @enum {number}
 */
export const TextureChannel = {
  R: 0,
  G: 1,
  B: 2,
  A: 3,
};

/**
 * Converts render pass flags to a string.
 * @param {number} flags - The flags to convert.
 * @returns {string} The string representation of the flags.
 */
export function render_pass_flags_to_string(flags) {
  const flag_names = [];
  if (flags & RenderPassFlags.Graphics) flag_names.push(graphics_pass_name);
  if (flags & RenderPassFlags.Present) flag_names.push(present_pass_name); 
  if (flags & RenderPassFlags.Compute) flag_names.push(compute_pass_name);
  if (flags & RenderPassFlags.GraphLocal) flag_names.push(graph_local_pass_name);
  return flag_names.join(", ");
}
