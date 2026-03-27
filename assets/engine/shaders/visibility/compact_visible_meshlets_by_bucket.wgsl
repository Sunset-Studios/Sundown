#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

struct VisibilityBucketInfo {
    current_visibility_bucket: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var<storage, read> object_instances: array<ObjectInstance>;
@group(1) @binding(1) var<storage, read> in_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(2) var<storage, read> in_draw_command: array<MeshletDrawCommandNoAtomics>;
@group(1) @binding(3) var<uniform> visibility_bucket_info: VisibilityBucketInfo;
@group(1) @binding(4) var<storage, read_write> out_visible_meshlets: array<vec4<u32>>;
@group(1) @binding(5) var<storage, read_write> out_draw_command: array<MeshletDrawCommand>;

@compute @workgroup_size(128)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let visible_index = gid.x;
    if (visible_index >= in_draw_command[0].instance_count || visible_index >= arrayLength(&in_visible_meshlets)) {
        return;
    }

    let visible_entry = in_visible_meshlets[visible_index];
    let object_instance_index = meshlet_object_index(visible_entry);
    if (object_instances[object_instance_index].visibility_bucket != visibility_bucket_info.current_visibility_bucket) {
        return;
    }

    let append_index = atomicAdd(&out_draw_command[0].instance_count, 1u);

    out_visible_meshlets[append_index] = visible_entry;
}
