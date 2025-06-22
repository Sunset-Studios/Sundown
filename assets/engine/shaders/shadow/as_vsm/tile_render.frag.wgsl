// AS-VSM Stage B: Render Shadow Casters into shadow atlas (Fragment)
// Renders geometry into each requested tile viewport.
#include "common.wgsl"
#include "lighting_common.wgsl"
#include "shadow/shadows_common.wgsl"

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_object_instances: array<i32>;
@group(1) @binding(3) var<uniform> vsm_settings: ASVSMSettings;
@group(1) @binding(4) var page_table: texture_storage_2d_array<r32uint, read>; // PTE format: Bit31=Valid, Bits30-27=LOD, Bits26-0=PhysID
@group(1) @binding(5) var<uniform> light_ub: ShadowCasterLight;
@group(1) @binding(6) var<storage, read> bitmask: array<u32>;
@group(1) @binding(7) var<storage, read> light_view_buffer: array<u32>;
@group(1) @binding(8) var<storage, read> light_shadow_idx_buffer: array<u32>;
@group(1) @binding(9) var<storage, read_write> shadow_atlas_depth: array<atomic<u32>>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) virtual_pixel: vec2<f32>,
  @location(1) @interpolate(flat) clipmap_index: u32,
  @location(2) @interpolate(flat) shadow_index: u32,
};

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
#if SHADOWS_ENABLED
  // Use interpolated virtual_pixel to calculate physical atlas coords
  let virtual_pixel_adjusted = input.virtual_pixel + vec2<f32>(0.5);
  let tile_xy_f     = virtual_pixel_adjusted / vsm_settings.tile_size;
  let tile_coords   = vec2<u32>(floor(tile_xy_f));

  // Page table lookup to find physical page location
  let slice_index   = input.clipmap_index + input.shadow_index * u32(vsm_settings.max_lods);
  let entry         = textureLoad(page_table, tile_coords, slice_index).r;

  // Discard fragments that fall into non-resident pages
  if (!vsm_pte_is_resident(entry)) {
    discard;
  }

  let physical_xy       = vsm_pte_get_phys_xy(entry);
  let memory_pool_index = vsm_pte_get_memory_pool_index(entry);

  let local_pixel       = vec2<u32>(fract(tile_xy_f) * vsm_settings.tile_size);
  let physical_pixel    = physical_xy * u32(vsm_settings.tile_size) + local_pixel;

  // Bounds check
  let phys_dim          = u32(vsm_settings.physical_dim);
  if (physical_pixel.x >= phys_dim || physical_pixel.y >= phys_dim) {
      return vec4<f32>(0.0);
  }
  
  // Calculate depth from fragment's interpolated position (map NDC [-1,1] to [0,1])
  let ndc_depth = input.position.z / input.position.w;
  let depth_01 = ndc_depth * 0.5 + 0.5; // Convert [-1,1] to [0,1]
  let depth_bits = pack_depth(depth_01);

  // Linear index:  slice_offset + y * width + x
  let slice_offset = memory_pool_index * phys_dim * phys_dim;
  let linear_index = slice_offset + physical_pixel.y * phys_dim + physical_pixel.x;

  // Atomically keep the smallest depth (closest fragment)
  atomicMin(&shadow_atlas_depth[linear_index], depth_bits);
#endif
  return vec4<f32>(0.0);
}