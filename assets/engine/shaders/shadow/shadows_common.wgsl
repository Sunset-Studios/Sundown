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
  max_tile_requests: f32,
  clip0_extent: f32,
};

// ------------------------------------------------------------------------------------
// Shadows
// ------------------------------------------------------------------------------------
#if SHADOWS_ENABLED
// ------------------------------------------------------------------
// Adaptive Sparse VSM – Page-table helpers (32-bit entry layout)
// ------------------------------------------------------------------
// Layout (least-significant bit first):
//  bits  0-6   : physical_x index (7-bit)
//  bits  7-13  : physical_y index (7-bit)
//  bits 14-16  : pool_id          (3-bit) – currently always 0
//  bit  17     : residency flag   (1-bit)
//  bit  18     : dirty flag       (1-bit – 1 == needs render)
//  bits 19-26  : frame_age        (8-bit)
//  bits 27-31  : reserved

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

// Convert phys_{x,y} to a linear id in row-major order
fn vsm_pte_get_physical_id(pte: u32, settings: ASVSMSettings) -> u32 {
    let tile_xy = vsm_pte_get_phys_xy(pte);
    let phys_tiles_per_row = u32(settings.physical_tiles_per_row);
    return tile_xy.y * phys_tiles_per_row + tile_xy.x;
}

// Virtual-tile index → (x, y, lod)
fn vsm_pte_get_tile_coords(virtual_tile_id: u32, settings: ASVSMSettings) -> vec3<u32> {
    let tile_per_row   = u32(settings.virtual_tiles_per_row);
    let tiles_per_lod  = tile_per_row * tile_per_row;
    let local_index    = virtual_tile_id % tiles_per_lod;
    let tile_x         = local_index % tile_per_row;
    let tile_y         = local_index / tile_per_row;
    let tile_lod       = virtual_tile_id / tiles_per_lod;
    return vec3<u32>(tile_x, tile_y, tile_lod);
}

// Convert clipmap 0 coordinates to clipmap n coordinates
fn vsm_convert_clip0_to_clipn(original: vec3<f32>, clip_map_index: u32) -> vec3<f32> {    
    let one_over_pow2 = 1.0 / f32(bitmask_pow2(clip_map_index));
    return vec3<f32>(original.x * one_over_pow2, original.y * one_over_pow2, original.z * one_over_pow2);                                                   
}

// Returns values on the range of [-1, 1]
fn vsm_calculate_render_clip_value_from_world_pos(world_pos: vec3<f32>, clip_map_index: u32, clipmap0_projection_view_render: mat4x4<f32>) -> vec3<f32> {
    let result = (clipmap0_projection_view_render * vec4<f32>(world_pos, 1.0));
    return vsm_convert_clip0_to_clipn(result.xyz, clip_map_index);
}

// Subtracts off the scaled translate component
fn vsm_calculate_sample_clip_value_from_world_pos(world_pos: vec3<f32>, clip_map_index: u32, clipmap0_projection_view: mat4x4<f32>) -> vec3<f32> {
    let result = vsm_calculate_render_clip_value_from_world_pos(world_pos, clip_map_index, clipmap0_projection_view);
    // Accounts for the fact that each clip map covers double the range of the previous
    return result - vsm_convert_clip0_to_clipn(clipmap0_projection_view[3].xyz, clip_map_index);
}

fn vsm_convert_world_pos_to_physical_coordinates(world_pos: vec3<f32>, clip_map_index: u32, clipmap0_projection_view: mat4x4<f32>, settings: ASVSMSettings, page_table: texture_storage_2d_array<r32uint, read>) -> vec3<f32> {
    let ndc = vsm_calculate_sample_clip_value_from_world_pos(world_pos, clip_map_index, clipmap0_projection_view);

    var virtual_tex_coords = ndc.xy * 0.5 + 0.5;

    virtual_tex_coords = fract(virtual_tex_coords);

    let page_table_index = vec2<u32>(floor(virtual_tex_coords * vec2<f32>(settings.virtual_tiles_per_row)));
    let entry = textureLoad(page_table, page_table_index, clip_map_index).x;

    let physical_page_xy_offset = vsm_pte_get_phys_xy(entry);
    let memory_pool_index = vsm_pte_get_memory_pool_index(entry);

    let virtual_pixel_coords = virtual_tex_coords * vec2<f32>(settings.tile_size);
    let physical_page_texel_offsets = vec2<f32>(
        virtual_pixel_coords.x % settings.tile_size,
        virtual_pixel_coords.y % settings.tile_size
    );

    return vec3<f32>(
        vec2<f32>(physical_page_xy_offset) * vec2<f32>(settings.tile_size) + physical_page_texel_offsets,
        f32(memory_pool_index)
    );
}

fn vsm_calculate_clipmap_index_from_world_pos(world_pos: vec3<f32>, clipmap0_projection_view_render: mat4x4<f32>) -> u32 {
    // Get the normalized device coordinates (NDC) for the first clipmap.
    let ndc = vsm_calculate_render_clip_value_from_world_pos(world_pos, 0, clipmap0_projection_view_render);

    let radius  = length(ndc.xyz);
    let lod     = ceil(log2(max(radius, 1.0)));

    return u32(lod);
}

#endif