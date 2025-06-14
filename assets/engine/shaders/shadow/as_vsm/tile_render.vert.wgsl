// AS-VSM Stage D: Render Shadow Casters into shadow atlas (Vertex)
// Renders geometry into each requested tile viewport.
#include "common.wgsl"
#include "lighting_common.wgsl"

// Add draw index uniform buffer for per-draw metadata indexing
struct ShadowCasterDrawIndexUniform {
    request_index: u32, 
};

@group(1) @binding(0) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(1) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(2) var<storage, read> visible_object_instances: array<i32>;
@group(1) @binding(3) var<storage, read> requested_tiles: array<u32>; // Format: [count, vt_id, view_mask, ...]
@group(1) @binding(4) var<storage, read> dense_shadow_casting_lights_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> settings: ASVSMSettings;
@group(1) @binding(6) var page_table: texture_storage_2d_array<r32uint, read>; // PTE format: Bit31=Valid, Bits30-27=LOD, Bits26-0=PhysID
@group(1) @binding(7) var<uniform> shadow_caster_draw_index_ub: ShadowCasterDrawIndexUniform;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32,
        @builtin(instance_index) instance_index: u32) -> VertexOutput {
  var out: VertexOutput;

  // Total number of active tile requests is stored at requested_tiles[0]
  let active_request_count = requested_tiles[0];

  // Skip rendering if this draw's request index is out of range
  if (shadow_caster_draw_index_ub.request_index >= active_request_count) {
    // Position the vertex outside the clip space to effectively discard it
    out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
  }

  let object_instance_index = visible_object_instances[instance_index];

  // Pull per-draw metadata using draw index uniform
  let request_index = 1u + shadow_caster_draw_index_ub.request_index * 3u;
  let tile_id = requested_tiles[request_index];
  let light_index = requested_tiles[request_index + 1u];
  let view_index = requested_tiles[request_index + 2u];
  let shadow_casting_light_index = dense_shadow_casting_lights_buffer[light_index];

  // decode virtual tile coords from the virtual tile_id
  let pte_coords = vsm_pte_get_tile_coords(tile_id, settings);

  // Fetch the PTE
  let pte_val = textureLoad(page_table, pte_coords.xy, shadow_casting_light_index).r;
  let phys_id = vsm_pte_get_physical_id(pte_val);

  // now remap into the *physical* atlas
  let phys_tiles_per_row  = u32(settings.physical_tiles_per_row); // float from settings
  let phys_x          = phys_id % phys_tiles_per_row;
  let phys_y          = phys_id / phys_tiles_per_row;

  let one_over_phys   = 1.0 / f32(phys_tiles_per_row);
  let offset_x        = (2.0 * f32(phys_x) + 1.0) * one_over_phys - 1.0;
  let offset_y        = (2.0 * f32(phys_y) + 1.0) * one_over_phys - 1.0;

  // Lookup model transform for this mesh instance
  let row_field = object_instances[object_instance_index].row; // Assuming 'row' contains info to get transform
  let true_row = get_entity_row(row_field); // Helper to decode row_field
  let transform = entity_transforms[true_row].transform;

  // Fetch local vertex position
  // Assuming a global vertex_buffer is available or accessed via batch_idx somehow
  let local_pos = vertex_buffer[vertex_index].position; // Placeholder if direct access isn't right

  // Compute world-space position
  let world_pos = transform * local_pos;

  // Get light view-projection matrix for this light index
  let light_vp = view_buffer[view_index].view_projection_matrix;

  // Transform into light clip space
  var clip = light_vp * world_pos;

  // Calculate tile viewport transform
  // Each tile should represent a portion of the light's view frustum
  // We only apply offsets to position the viewport, no scaling
  let tile_size = 2.0 / f32(phys_tiles_per_row);  // Size of each tile in clip space
  let tile_offset_x = f32(phys_x) * tile_size - 1.0;  // Offset from left edge of viewport
  let tile_offset_y = f32(phys_y) * tile_size - 1.0;  // Offset from bottom edge of viewport

  // Only apply the offset to position the viewport
  clip = vec4<f32>(
    clip.x + tile_offset_x,
    clip.y + tile_offset_y,
    clip.z,
    clip.w
  );

  out.position = clip;

  return out;
}

// TODO: Adding a fragment shader that writes depth to the shadow atlas but also writes the view id to a per-pixel linked list.
