// AS-VSM Stage C: New-tile Gather
// Scans bitmask and identifies tiles that are set but not resident.
#include "common.wgsl"
#include "lighting_common.wgsl"

@group(1) @binding(0) var<storage, read> bitmask: array<u32>;
@group(1) @binding(1) var page_table: texture_storage_2d_array<r32uint, read>;
@group(1) @binding(2) var<storage, read_write> requested_tiles: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> dense_lights_buffer: array<Light>;
@group(1) @binding(4) var<storage, read> dense_shadow_casting_lights_buffer: array<u32>;
@group(1) @binding(5) var<storage, read> light_count_buffer: array<u32>;
@group(1) @binding(6) var<storage, read> settings: ASVSMSettings;

// Atomic counter for append index (stored at requested_tiles[0])
// Assumes requested_tiles[0] is initialized to 0 before dispatch

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    // Compute linear index into bitmask array
    let index = id.y * 8u + id.x;
    let count = arrayLength(&bitmask);
    if (index >= count) {
        return;
    }

    let light_count = light_count_buffer[0u];
    if (id.z >= light_count) {
        return;
    }

    // Fetch mask of virtual tiles
    var bits = bitmask[index];

    // Capture the light/view index from the dispatch.z
    let dense_light_index = id.z;
    let shadow_casting_light_index = dense_shadow_casting_lights_buffer[dense_light_index];

    let light = dense_lights_buffer[dense_light_index];
    if ((light.activated == 0.0) || (light.shadow_casting == 0.0)) {
        return;
    }

    let view_index = u32(light.view_index);

    while(bits != 0u) {
      // Find least significant set bit
      let shift = countTrailingZeros(bits);
      let tile_id = index * 32u + shift;

      // Decode virtual tile coordinates and check if PTE is valid
      let pte_coords = vsm_pte_get_tile_coords(tile_id, settings);
      let pte_val = textureLoad(page_table, pte_coords.xy, shadow_casting_light_index).r;
      let pte_is_valid = vsm_pte_is_valid(pte_val);
      
      // If not valid, request it
      if (!pte_is_valid) {
        let max_tile_requests = u32(settings.max_tile_requests);

        // try to reserve one slot, but never go past max_tile_requests
        var slot: u32;
        loop {
          let current_count = atomicLoad(&requested_tiles[0]);
          // no more slots available?
          if (current_count >= max_tile_requests) {
            break;
          }
          // attempt to bump current_count → current_count+1
          let compare_result = atomicCompareExchangeWeak(
            &requested_tiles[0],
            current_count,
            current_count + 1u
          );
          if (compare_result.exchanged) {
            slot = current_count;
            // we successfully reserved slot "slot"
            let base = 1u + slot * 3u;
            atomicStore(&requested_tiles[base],        tile_id);
            atomicStore(&requested_tiles[base + 1u], dense_light_index);
            atomicStore(&requested_tiles[base + 2u], view_index);
            break;
          }
          // else: another thread won the race, retry
        }
      }

      bits = bits & (bits - 1u); // Clear LSB
    }
} 