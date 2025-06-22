// ------------------------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------------------------

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
};

struct VirtualTileInfo {
    tile_coords: vec2<u32>,
    tile_xy_f: vec2<f32>,
    clipmap_index: u32,
    virtual_pixel: vec2<f32>,
};

struct PhysicalTileInfo {
    local_pixel: vec2<f32>,
    physical_pixel: vec2<f32>,
    physical_xy: vec2<u32>,
    memory_pool_index: u32,
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

// ------------------------------------------------------------------
fn bitmask_pow2(shift: u32) -> u32 {
    return 1u << shift;
}

fn vsm_get_virtual_tile_word_and_mask(tile_coords: vec2<u32>, clipmap_index: u32, shadow_index: u32, settings: ASVSMSettings) -> vec2<u32> {
  let vtr             = u32(settings.virtual_tiles_per_row);
  let base_index      = clipmap_index * vtr * vtr;
  let tile_id         = base_index + tile_coords.y * vtr + tile_coords.x;

  let word_index = tile_id >> 5u;
  let bit_index  = tile_id & 31u;
  let mask       = 1u << bit_index;

  // Compute per-light stride so each light writes to its own range
  let stride_words = ((vtr * vtr * u32(settings.max_lods) + 31u) >> 5u);
  let global_word_index = shadow_index * stride_words + word_index;
  
  return vec2<u32>(global_word_index, mask);
}

fn vsm_pte_is_resident(pte: u32) -> bool {
    return (pte & pte_residency_mask) != 0u;
}

fn vsm_pte_is_dirty(pte: u32) -> bool {
    return (pte & pte_dirty_mask) != 0u;
}

// A page is "valid" when it is resident *and* not marked dirty
fn vsm_pte_is_valid(pte: u32) -> bool {
    return vsm_pte_is_resident(pte) && !vsm_pte_is_dirty(pte);
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
    let vtl            = vtr * vtr;
    let local_index    = virtual_tile_id % vtl;
    let tile_x         = local_index % vtr;
    let tile_y         = local_index / vtr;
    let tile_lod       = virtual_tile_id / vtl;
    return vec3<u32>(tile_x, tile_y, tile_lod);
}

// Convert clipmap 0 coordinates to clipmap n coordinates
fn vsm_convert_clip0_to_clipn(original: vec4<f32>, clip_map_index: u32) -> vec4<f32> {    
    let one_over_pow2 = 1.0 / f32(bitmask_pow2(clip_map_index));
    return vec4<f32>(original.xy * one_over_pow2, original.z, original.w);                                                   
}

// Returns values on the range of [-1, 1]
fn vsm_calculate_render_clip_value_from_world_pos(world_pos: vec4<f32>, clip_map_index: u32, clipmap0_projection_view_render: mat4x4<f32>) -> vec4<f32> {
    var result = clipmap0_projection_view_render * world_pos;
    return vsm_convert_clip0_to_clipn(result, clip_map_index);
}

// Subtracts off the scaled translate component
fn vsm_calculate_sample_clip_value_from_world_pos(world_pos: vec4<f32>, clip_map_index: u32, clipmap0_projection_view: mat4x4<f32>) -> vec4<f32> {
    let result = vsm_calculate_render_clip_value_from_world_pos(world_pos, clip_map_index, clipmap0_projection_view);
    let vp_clip = vsm_convert_clip0_to_clipn(clipmap0_projection_view[3], clip_map_index) / result.w;
    // Accounts for the fact that each clip map covers double the range of the previous
    return result - vp_clip;
}

// Virtual-tile → Physical-tile
fn vsm_vtile_to_ptile(vtile_info: VirtualTileInfo, settings: ASVSMSettings, page_table: texture_storage_2d_array<r32uint, read>) -> PhysicalTileInfo {
    var info: PhysicalTileInfo;
    let entry = textureLoad(page_table, vtile_info.tile_coords, vtile_info.clipmap_index).r;
    info.physical_xy = vsm_pte_get_phys_xy(entry);
    info.memory_pool_index = vsm_pte_get_memory_pool_index(entry);
    info.local_pixel = fract(vtile_info.tile_xy_f) * settings.tile_size;
    info.physical_pixel = vec2<f32>(info.physical_xy) * settings.tile_size + info.local_pixel;
    return info;
}

fn vsm_calculate_clipmap_index_from_world_pos(world_pos: vec4<f32>, clipmap0_projection_view_render: mat4x4<f32>) -> u32 {
    // Get the normalized device coordinates (NDC) for the first clipmap.
    var ndc = vsm_calculate_render_clip_value_from_world_pos(world_pos, 0, clipmap0_projection_view_render);

    let radius  = length(ndc.xyz);
    let lod     = ceil(log2(max(radius, 1.0)));

    return u32(lod);
}

fn vsm_world_to_virtual_tile(world_pos: vec4<f32>, camera_vp: mat4x4<f32>, clipmap0_vp: mat4x4<f32>, settings: ASVSMSettings) -> VirtualTileInfo {
    var info: VirtualTileInfo;

    info.clipmap_index = clamp(
        vsm_calculate_clipmap_index_from_world_pos(world_pos, camera_vp),
        0u,
        u32(settings.max_lods) - 1u
    );

    let sample_clip   = vsm_calculate_sample_clip_value_from_world_pos(world_pos, info.clipmap_index, clipmap0_vp);
    let uv_full       = sample_clip.xy * 0.5 + 0.5;
    info.virtual_pixel = fract(uv_full) * settings.virtual_dim;

    info.tile_xy_f     = (info.virtual_pixel + vec2<f32>(0.5)) / settings.tile_size;
    info.tile_coords   = vec2<u32>(floor(info.tile_xy_f));

    return info;
}

fn vsm_tile_transform(scaled_pos: vec4<f32>, tile_xy_f: vec2<f32>, pte: u32, settings: ASVSMSettings) -> vec4<f32> {
    let local_clip = fract(tile_xy_f) * 2.0 - 1.0;

    return vec4<f32>(
        local_clip.x * scaled_pos.w,
        local_clip.y * scaled_pos.w,
        scaled_pos.z,
        scaled_pos.w
    );
}

#endif