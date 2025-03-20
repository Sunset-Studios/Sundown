#include "common.wgsl"

@group(1) @binding(0) var<storage, read_write> entity_positions: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> entity_flags: array<u32>;

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entity = global_id.x;
    if (entity >= arrayLength(&entity_flags)) {
        return;
    }

    let seed = f32(random_seed(u32(entity))) * 0.0001;
    let time = (frame_info.time + seed) * 2.0;

    var entity_position = entity_positions[entity];

    entity_position.y += sin(time) * 0.02;

    entity_positions[entity] = entity_position;
}