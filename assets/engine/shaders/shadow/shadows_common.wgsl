// ------------------------------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------------------------------ 

struct ASVSMSettings {
  tile_size: f32,
  virtual_dim: f32,
  virtual_tiles_per_row: f32,
  physical_dim: f32,
  physical_tiles_per_row: f32,
  max_lods: f32,
  max_physical_pools: f32,
  clip0_extent: f32,
};

struct ShadowCasterLight {
    light_index: u32,
    clip_index: u32,
};

struct VirtualTileInfo {
    tile_coords: vec2<u32>,
    local_pixel: vec2<u32>,
    clipmap_index: u32,
    tile_id: u32,
};

struct PhysicalTileInfo {
    physical_pixel: vec2<u32>,
    physical_xy: vec2<u32>,
    physical_id: u32,
    memory_pool_index: u32,
    is_resident: bool,
    is_dirty: bool,
};

struct ShadowFilterResult {
    depth: f32, // bilinear-filtered depth sample (range 0-1)
    valid: bool, // whether the underlying physical page is resident & clean
};

// ------------------------------------------------------------------------------------
// Shadows
// ------------------------------------------------------------------------------------
#if SHADOWS_ENABLED
// ------------------------------------------------------------------
// Adaptive Sparse VSM – Page-table helpers (32-bit entry layout)
// ------------------------------------------------------------------

// =============================================================
//  Page-table entry (32-bit) bit-field layout
//  [0 – 6]   : physical page X index   (7 bits)  (0-127)
//  [7 – 13]  : physical page Y index   (7 bits)  (0-127)
//  [14 – 16] : atlas / memory-pool id  (3 bits)  (0-7)
//  [17]      : residency flag          (1 bit)   (1 = resident)
//  [18]      : dirty flag              (1 bit)   (1 = needs update)
//  [19 – 26] : frame age / marker      (8 bits)  (wraps every 256 frames)
//  [27 – 31] : reserved / unused
// =============================================================

// Masks / shifts (snake_case)
const pte_phys_x_shift          : u32 = 0u;
const pte_phys_x_mask           : u32 = 0x0000007Fu;

const pte_phys_y_shift          : u32 = 7u;
const pte_phys_y_mask           : u32 = 0x00003F80u;

const pte_pool_id_shift         : u32 = 14u;
const pte_pool_id_mask          : u32 = 0x0001C000u;

const pte_residency_shift       : u32 = 17u;
const pte_residency_mask        : u32 = 0x00020000u;

const pte_dirty_shift           : u32 = 18u;
const pte_dirty_mask            : u32 = 0x00040000u;

const pte_frame_age_shift       : u32 = 19u;
const pte_frame_age_mask        : u32 = 0x07F80000u;

const pte_dirty_linger_shift    : u32 = 27u;
const pte_dirty_linger_mask     : u32 = 0x78000000u;

const vsm_dirty_linger_frames   : u32 = 1u;

const lru_pinned_flag           : u32 = 0x80000000u;

// C is a constant that controls the log depth curve; try 1000.0 or scene far/near ratio
const LOG_C = 2000.0;

// Pack log depth
fn pack_log_depth(depth: f32) -> f32 {
    return log(depth * LOG_C + 1.0) / log(LOG_C + 1.0);
}

// Unpack log depth
fn unpack_log_depth(packed: f32) -> f32 {
    return (exp(packed * log(LOG_C + 1.0)) - 1.0) / LOG_C;
}

// ------------------------------------------------------------------
// Packs clip-space depth (range [0,1]) into an unsigned 32-bit integer such that
// larger integers correspond to *nearer* fragments for reverse Z.
// This makes it compatible with atomicMax for closest-depth selection.
fn pack_depth(clip_depth: f32) -> u32 {
    // For reverse Z: 1.0 = near, 0.0 = far
    // We want larger packed values for nearer fragments (for atomicMax)
    let reversed_depth = clip_depth;
    return u32(reversed_depth * 4294967295.0);
}

// Converts a packed depth integer back to clip-space depth in [0,1] for reverse Z.
// The caller can further convert to linear eye-space depth via linearize_depth.
fn unpack_depth(packed_depth: u32) -> f32 {
    let reversed_depth = f32(packed_depth) / (4294967295.0);
    return reversed_depth;
}

fn bitmask_pow2(shift: u32) -> u32 {
    return 1u << shift;
}

fn vsm_pte_is_resident(pte: u32) -> bool {
    return (pte & pte_residency_mask) != 0u;
}

fn vsm_pte_is_dirty(pte: u32) -> bool {
    return (pte & pte_dirty_mask) != 0u;
}

fn vsm_pte_get_frame_age(pte: u32) -> u32 {
    return ((pte & pte_frame_age_mask) >> pte_frame_age_shift) & 0x000000FFu;
}

fn vsm_pte_get_dirty_linger(pte: u32) -> u32 {
    return (pte & pte_dirty_linger_mask) >> pte_dirty_linger_shift;
}

fn vsm_pte_set_dirty_linger(pte: u32, linger: u32) -> u32 {
    let clamped_linger = min(linger, 0xFu);
    return (pte & ~pte_dirty_linger_mask) |
        ((clamped_linger << pte_dirty_linger_shift) & pte_dirty_linger_mask);
}

fn vsm_pte_mark_dirty(pte: u32) -> u32 {
    return vsm_pte_set_dirty_linger(pte | pte_dirty_mask, vsm_dirty_linger_frames);
}

// A page is "valid" when it is resident *and* not marked dirty
fn vsm_pte_is_valid(pte: u32) -> bool {
    return vsm_pte_is_resident(pte);
}

// Extract (phys_x, phys_y) from the entry
fn vsm_pte_get_phys_xy(pte: u32) -> vec2<u32> {
    let phys_x = (pte & pte_phys_x_mask) >> pte_phys_x_shift;
    let phys_y = (pte & pte_phys_y_mask) >> pte_phys_y_shift;
    return vec2<u32>(phys_x, phys_y);
}

// Extract memory pool index from the entry
fn vsm_pte_get_memory_pool_index(pte: u32) -> u32 {
    return (pte & pte_pool_id_mask) >> pte_pool_id_shift;
}

// Virtual-tile index → (x, y, lod)
fn vsm_pte_get_tile_coords(virtual_tile_id: u32, settings: ASVSMSettings) -> vec3<u32> {
    let vtr            = u32(settings.virtual_tiles_per_row);
    let vtc            = vtr * vtr;
    let tile_clip      = virtual_tile_id / vtc;
    let local_index    = virtual_tile_id % vtc;
    let tile_x         = local_index % vtr;
    let tile_y         = local_index / vtr;
    return vec3<u32>(tile_x, tile_y, tile_clip);
}

// Physical ID helpers – convert between physical IDs and XY pool IDs.
fn vsm_physical_id_to_xy_pool(physical_id: u32, settings: ASVSMSettings) -> vec3<u32> {
  let ptpr = u32(settings.physical_tiles_per_row);
  let atlas_size = ptpr * ptpr;
  let pool_id = physical_id / atlas_size;
  let local_physical_id = physical_id % atlas_size;
  let phys_x = local_physical_id % ptpr;
  let phys_y = local_physical_id / ptpr;
  return vec3<u32>(phys_x, phys_y, pool_id);
}

// PTE helpers – convert between PTE values and physical IDs.
fn vsm_pte_to_physical_id(pte_entry: u32, settings: ASVSMSettings) -> u32 {
  let ptpr = u32(settings.physical_tiles_per_row);
  let atlas_size = ptpr * ptpr;
  let pool_id = (pte_entry & pte_pool_id_mask) >> pte_pool_id_shift;
  let phys_x = (pte_entry & pte_phys_x_mask) >> pte_phys_x_shift;
  let phys_y = (pte_entry & pte_phys_y_mask) >> pte_phys_y_shift;
  return pool_id * atlas_size + phys_y * ptpr + phys_x;
}

// Convert clip0 to clipn
fn vsm_convert_clip0_to_clipn(original : vec4<f32>,
                              clip_map_index : u32,
                              settings: ASVSMSettings) -> vec4<f32> {
    let one_over_pow2 = 1.0 / f32(bitmask_pow2(clip_map_index));
    return vec4<f32>(original.x * one_over_pow2,
                     original.y * one_over_pow2,
                     original.z,
                     original.w);
}

fn vsm_projection_translation_clip(
    clipmap0_projection_view: mat4x4<f32>,
    clip_map_index: u32,
    settings: ASVSMSettings
) -> vec4<f32> {
    return vsm_convert_clip0_to_clipn(
        vec4<f32>(clipmap0_projection_view[3].xyz, 1.0),
        clip_map_index,
        settings
    );
}

// Returns values on the range of [-1, 1]
fn vsm_calculate_render_clip_value_from_world_pos(
    world_pos: vec4<f32>,
    clip_map_index: u32,
    clipmap0_projection_view_render: mat4x4<f32>,
    settings: ASVSMSettings
) -> vec4<f32> {
    // Project to clip space and perform perspective divide to get normalized device coordinates
    var clip = clipmap0_projection_view_render * world_pos;
    clip /= clip.w;
    return vsm_convert_clip0_to_clipn(clip, clip_map_index, settings);
}

// Subtracts off the scaled translate component
fn vsm_calculate_sample_clip_value_from_world_pos(
    world_pos: vec4<f32>,
    clip_map_index: u32,
    clipmap0_projection_view: mat4x4<f32>,
    settings: ASVSMSettings
) -> vec4<f32> {
    let result = vsm_calculate_render_clip_value_from_world_pos(
        world_pos,
        clip_map_index,
        clipmap0_projection_view,
        settings
    );
    return result - vsm_snapped_translation_for_lod(clipmap0_projection_view, clip_map_index, settings);
}

fn vsm_calculate_clipmap_index_from_world_pos(
    world_pos: vec4<f32>,
    camera_vp: mat4x4<f32>,
    settings: ASVSMSettings
) -> u32 {
    var clip      = camera_vp * world_pos;
    let radius    = length(clip.xyz);
    let lod       = floor(log2(max(radius, 1.0)));
    return u32(lod);
}

fn vsm_world_to_virtual_tile(
    world_pos: vec4<f32>,
    camera_vp: mat4x4<f32>,
    clipmap0_vp: mat4x4<f32>,
    settings: ASVSMSettings
) -> VirtualTileInfo {
    var clipmap_index = vsm_calculate_clipmap_index_from_world_pos(world_pos, camera_vp, settings);
    clipmap_index = clamp(clipmap_index, 0u, u32(settings.max_lods) - 1u);

    let vtr = u32(settings.virtual_tiles_per_row);
    let tile_size = u32(settings.tile_size);

    var sample_clip    = vsm_calculate_sample_clip_value_from_world_pos(
                            world_pos,
                            clipmap_index,
                            clipmap0_vp,
                            settings
                        );
    let virtual_uv        = fract(sample_clip.xy * 0.5 + 0.5);
    // Wrap to the clipmap range and offset by half a texel for stable mapping
    let virtual_pixel    = vec2<u32>(virtual_uv * settings.virtual_dim);

    var info: VirtualTileInfo;
                        
    info.clipmap_index    = clipmap_index;
    info.local_pixel      = virtual_pixel % tile_size;
    info.tile_coords      = virtual_pixel / tile_size;
    info.tile_id          = info.clipmap_index * vtr * vtr + info.tile_coords.y * vtr + info.tile_coords.x;

    return info;
}

// Variant of vsm_world_to_virtual_tile that uses a caller-supplied
// clipmap_index. This is used when the clipmap being rendered is known a-priori
// (e.g. when doing one render pass per clipmap level) so we avoid per-vertex
// divergence and ensure all vertices of a primitive map to the same clipmap.
fn vsm_world_to_virtual_tile_for_clip(
    world_pos: vec4<f32>,
    clipmap0_vp: mat4x4<f32>,
    settings: ASVSMSettings,
    clipmap_index: u32,
) -> VirtualTileInfo {
    let vtr = u32(settings.virtual_tiles_per_row);
    let tile_size = u32(settings.tile_size);

    var sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
        world_pos,
        clipmap_index,
        clipmap0_vp,
        settings,
    );
    let virtual_uv        = fract(sample_clip.xy * 0.5 + 0.5);
    // Wrap to the clipmap range and offset by half a texel for stable mapping
    let virtual_pixel    = vec2<u32>(virtual_uv * settings.virtual_dim);

    var info: VirtualTileInfo;

    info.clipmap_index    = clipmap_index;
    info.local_pixel      = virtual_pixel % tile_size;
    info.tile_coords      = virtual_pixel / tile_size;
    info.tile_id          = info.clipmap_index * vtr * vtr + info.tile_coords.y * vtr + info.tile_coords.x;

    return info;
}

// Fetches the tile coords for a world position without wrapping.
// Specifically useful for the cull_shadows compute shader.
fn vsm_world_to_unwrapped_tile_coords(
    world_pos: vec4<f32>,
    clipmap0_vp: mat4x4<f32>,
    settings: ASVSMSettings,
    clipmap_index: u32,
) -> vec2<i32> {
    var sample_clip = vsm_calculate_sample_clip_value_from_world_pos(
        world_pos,
        clipmap_index,
        clipmap0_vp,
        settings,
    );
    let virtual_uv     = sample_clip.xy * 0.5 + 0.5;
    let tile_coords    = vec2<i32>((virtual_uv * settings.virtual_dim) / settings.tile_size);

    return tile_coords;
}

// Convert virtual tile to physical tile
fn vsm_vtile_to_ptile(
    vtile_info: VirtualTileInfo,
    settings: ASVSMSettings,
    shadow_index: u32,
    page_table: texture_storage_2d_array<r32uint, read>,
) -> PhysicalTileInfo {
    var info: PhysicalTileInfo;

    let entry              = textureLoad(
        page_table,
        vtile_info.tile_coords,
        vtile_info.clipmap_index + shadow_index * u32(settings.max_lods)
    ).r;
    let phys_dim = u32(settings.physical_dim);
    let tile_size = u32(settings.tile_size);

    info.is_resident       = vsm_pte_is_resident(entry);
    info.is_dirty          = vsm_pte_is_dirty(entry);
    info.physical_xy       = vsm_pte_get_phys_xy(entry);
    info.memory_pool_index = vsm_pte_get_memory_pool_index(entry);

    info.physical_pixel    = info.physical_xy * tile_size + vtile_info.local_pixel;
    info.physical_id       = info.memory_pool_index * phys_dim * phys_dim
        + info.physical_pixel.y * phys_dim + info.physical_pixel.x;
    return info;
}

// Get bitmask word and mask for a virtual tile
fn vsm_get_virtual_tile_word_and_mask(tile_coords: vec2<u32>, clipmap_index: u32, shadow_index: u32, settings: ASVSMSettings) -> vec2<u32> {
  let vtr             = u32(settings.virtual_tiles_per_row);
  let tiles_per_light = vtr * vtr * u32(settings.max_lods);
  let words_per_light = ((tiles_per_light + 31u) >> 5u);

  let base_index      = clipmap_index * vtr * vtr;
  let tile_id         = base_index + tile_coords.y * vtr + tile_coords.x;

  let word_index      = tile_id >> 5u;
  let bit_index       = tile_id & 31u;
  let mask            = 1u << bit_index;

  // Compute per-light stride so each light writes to its own range
  let global_word_index = shadow_index * words_per_light + word_index;
  
  return vec2<u32>(global_word_index, mask);
}

// Example usage inside vsm_snapped_translation_for_lod (or elsewhere):
fn vsm_snapped_translation_for_lod(
    clipmap0_vp : mat4x4<f32>,
    lod          : u32,
    settings     : ASVSMSettings
) -> vec4<f32> {
    // TODO: This might be a spot to try and properly snap the translation to the nearest page table size
    return vsm_projection_translation_clip(clipmap0_vp, lod, settings);
}

#endif