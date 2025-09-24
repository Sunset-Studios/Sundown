// ============================================================================
// BLAS Path Tracer Helpers (Per-Instance Triangle Hit)
// - Uses TLAS candidates to test per-mesh BLAS in local space
// - Strictly formatting and documentation improvements; no behavior changes
// ============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"

// ----------------------------------------------------------------------------
// Parameters and Per-Pixel Accumulation
// ----------------------------------------------------------------------------
struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    max_spp: u32,
};

struct PixelInfo {
    position_and_rng: vec4<f32>,
    normal_and_sample_count: vec4<f32>,
    accum_color: vec4<f32>,
};

// ----------------------------------------------------------------------------
// TLAS Candidate Record (input for BLAS pass)
//  - t_bits:       ieee-754 bits of t (bitcast from f32)
//  - prim_id:      resolved entity index (TLAS leaf payload)
//  - mesh_asset_id: mesh id for the primitive
// ----------------------------------------------------------------------------
struct TlasHit {
    t_bits: u32,
    prim_id: u32,
    mesh_asset_id: u32,
    pad1: u32,
};

// Number of TLAS candidates to consider per pixel
const TLAS_CANDIDATES: u32 = 4u;

// ----------------------------------------------------------------------------
// Bindings (group 1)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;              // Path tracer control params
@group(1) @binding(1) var<storage, read_write> pixel_info: array<PixelInfo>;   // Per-pixel accumulation
@group(1) @binding(2) var<storage, read> blas_bvh2_bounds: array<AABB>;        // BLAS leaf bounds
@group(1) @binding(3) var<storage, read> blas_bvh4_nodes: array<BVH4Node>;     // BLAS BVH4 nodes
@group(1) @binding(4) var<storage, read> tlas_hits: array<TlasHit>;            // TLAS candidates (from TLAS pass)
@group(1) @binding(5) var<storage, read> blas_directory: array<MeshDirectoryEntry>; // Mesh directory
@group(1) @binding(6) var<storage, read> entity_transforms: array<EntityTransform>; // Per-entity transforms
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;             // Triangle index buffer
@group(1) @binding(8) var position_tex: texture_2d<f32>;                       // GBuffer world position
@group(1) @binding(9) var normal_tex: texture_2d<f32>;                          // GBuffer world normal
@group(1) @binding(10) var output_tex: texture_storage_2d<rgba16float, write>;  // Output accumulation

// ----------------------------------------------------------------------------
// Debug: generate a stable color per primitive id
// ----------------------------------------------------------------------------
fn primitive_debug_color(mesh_asset_id: u32, prim_id: u32) -> vec3f {
    let seed0 = hash(mesh_asset_id ^ (prim_id * 0x9e3779b9u));
    let seed1 = hash(seed0 ^ 0x85ebca6bu);
    let seed2 = hash(seed1 ^ 0xc2b2ae35u);
    let r = f32(seed0) * one_over_float_max;
    let g = f32(seed1) * one_over_float_max;
    let b = f32(seed2) * one_over_float_max;
    return vec3f(r, g, b);
}

// ----------------------------------------------------------------------------
// Helper: Build a ray in mesh-local space from a world-space ray
// - Preserves tmin/tmax; returns fully-populated Ray with inv_direction
// ----------------------------------------------------------------------------
fn build_local_ray(ray_world: ptr<function, Ray>, entity_transform: ptr<function, mat4x4f>) -> Ray {
    let inv_m = inverse4x4(*entity_transform);
    let ro_world = (*ray_world).origin_and_tmin.xyz;
    let rd_world = (*ray_world).direction_and_tmax.xyz;

    var ray_local: Ray;
    ray_local.origin_and_tmin = vec4f((inv_m * vec4f(ro_world, 1.0)).xyz, (*ray_world).origin_and_tmin.w);
    ray_local.direction_and_tmax = vec4f((inv_m * vec4f(rd_world, 0.0)).xyz, (*ray_world).direction_and_tmax.w);

    let d = ray_local.direction_and_tmax.xyz;
    ray_local.inv_direction = vec4f(
        1.0 / max(abs(d.x), 1e-8) * select(1.0, -1.0, d.x < 0.0),
        1.0 / max(abs(d.y), 1e-8) * select(1.0, -1.0, d.y < 0.0),
        1.0 / max(abs(d.z), 1e-8) * select(1.0, -1.0, d.z < 0.0),
        0.0
    );
    return ray_local;
}

fn trace_blas(ray_world: ptr<function, Ray>, entity_transform: ptr<function, mat4x4f>, mesh_asset_id: u32) -> RayHit {
    let mesh_directory_entry = blas_directory[mesh_asset_id];
    let bvh2_base = mesh_directory_entry.bvh2_base;
    let bvh4_base = mesh_directory_entry.bvh4_base;
    let first_vertex = mesh_directory_entry.first_vertex;
    let first_index = mesh_directory_entry.first_index;

    // Build local ray from world ray and entity transform
    var ray_local = build_local_ray(ray_world, entity_transform);

    var hit: RayHit;
    hit.position_and_t = vec4f((*ray_world).origin_and_tmin.xyz, (*ray_world).direction_and_tmax.w);
    hit.normal_and_user_data = vec4f(0.0, 0.0, 0.0, -1.0);

    var node_stack: array<u32, 32>;
    node_stack[0] = bvh4_base;
    var stack_size = 1u;
    
    loop {
        if (stack_size == 0u) { break; }
        stack_size = stack_size - 1u;

        var node_idx = node_stack[stack_size];
        if (node_idx == INVALID_IDX) { continue; }

        let node = blas_bvh4_nodes[node_idx];
        let t_aabb = intersect_aabb(ray_local, node.min.xyz, node.max.xyz);

        if (t_aabb >= ray_local.origin_and_tmin.w && t_aabb < hit.position_and_t.w) {
            if (stack_size < 32u) {
                var min_index = 0u;
                var min_t = 1e38;

                let leaf_mask = bitcast<u32>(node.min.w);

                for (var i = 0u; i < 4u; i = i + 1u) {
                    let child_raw = node.children[i];
                    if (child_raw < 0.0) { continue; }

                    let is_leaf_child = ((leaf_mask >> i) & 1u) != 0u;
                    let child_idx = u32(child_raw);

                    if (is_leaf_child) {
                        let leaf_bounds = blas_bvh2_bounds[child_idx];
                        let t_leaf = intersect_aabb(ray_local, leaf_bounds.min.xyz, leaf_bounds.max.xyz);
                        if (t_leaf >= ray_local.origin_and_tmin.w && t_leaf < hit.position_and_t.w) {
                            // Triangle intersection in local space
                            let tri_id_local = u32(leaf_bounds.min.w);
                            let i0 = index_buffer[first_index + tri_id_local * 3u + 0u];
                            let i1 = index_buffer[first_index + tri_id_local * 3u + 1u];
                            let i2 = index_buffer[first_index + tri_id_local * 3u + 2u];

                            let v0 = vertex_buffer[first_vertex + i0].position.xyz;
                            let v1 = vertex_buffer[first_vertex + i1].position.xyz;
                            let v2 = vertex_buffer[first_vertex + i2].position.xyz;

                            let t_tri = intersect_triangle(ray_local, v0, v1, v2);
                            if (t_tri >= ray_local.origin_and_tmin.w && t_tri < hit.position_and_t.w) {
                                let p_local = ray_local.origin_and_tmin.xyz + ray_local.direction_and_tmax.xyz * t_tri;
                                let p_world = (*entity_transform * vec4f(p_local, 1.0)).xyz;
                                hit.position_and_t = vec4<f32>(p_world, t_tri);
                                hit.normal_and_user_data = vec4<f32>(0.0, 0.0, 0.0, f32(tri_id_local));
                            }
                        }
                    } else {
                        let child_node = blas_bvh4_nodes[child_idx];
                        let t_aabb_child = intersect_aabb(ray_local, child_node.min.xyz, child_node.max.xyz);

                        if (t_aabb_child >= ray_local.origin_and_tmin.w && t_aabb_child < hit.position_and_t.w) {
                            min_index = stack_size;
                            min_t = t_aabb_child;
                            node_stack[stack_size] = child_idx;
                            stack_size = stack_size + 1u;
                        }
                    }
                }

                let tmp_node = node_stack[min_index];
                node_stack[min_index] = node_stack[stack_size - 1u];
                node_stack[stack_size - 1u] = tmp_node;
            }
        }
    }

    return hit;
}

// ----------------------------------------------------------------------------
// Compute Kernel: Resolve TLAS candidates against BLAS and accumulate color
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(position_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = gid.y * res.x + gid.x;

    if (pt_params.reset_accum_flag != 0u) {
        pixel_info[pixel_index].accum_color = vec4f(0.0);
        pixel_info[pixel_index].normal_and_sample_count.w = 0.0;
        textureStore(output_tex, vec2<i32>(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let prev_count = pixel_info[pixel_index].normal_and_sample_count.w;

    if (pt_params.max_spp != 0u && prev_count >= f32(pt_params.max_spp)) {
        let prev_sum = pixel_info[pixel_index].accum_color;
        let denom = max(prev_count, 1.0);
        let avg = vec4f(prev_sum.xyz / denom, 1.0);
        textureStore(output_tex, vec2<i32>(gid.xy), avg);
        return;
    }

    let world_pos = textureLoad(position_tex, vec2<i32>(gid.xy), 0).xyz;
    let world_norm = textureLoad(normal_tex, vec2<i32>(gid.xy), 0).xyz;

    // Use the surface normal as a stable probe direction for BLAS sampling
    var ray_dir = safe_normalize(world_norm);
    var ray: Ray;
    ray.origin_and_tmin = vec4f(world_pos + ray_dir * 0.001, 0.001);
    ray.direction_and_tmax = vec4f(ray_dir, 1e30);
    ray.inv_direction = vec4f(1.0 / max(abs(ray_dir.x), 1e-8) * select(1.0, -1.0, ray_dir.x < 0.0),
                              1.0 / max(abs(ray_dir.y), 1e-8) * select(1.0, -1.0, ray_dir.y < 0.0),
                              1.0 / max(abs(ray_dir.z), 1e-8) * select(1.0, -1.0, ray_dir.z < 0.0),
                              0.0);

    let base = pixel_index * TLAS_CANDIDATES;
    var sample_rgb = vec3f(0.0);
    let original_tmin = ray.origin_and_tmin.w;
    for (var k = 0u; k < TLAS_CANDIDATES; k = k + 1u) {
        ray.origin_and_tmin.w = original_tmin;
        let hit_rec = tlas_hits[base + k];
        let entity_resolved = hit_rec.prim_id;
        if (entity_resolved == INVALID_IDX) { break; }
        
        let mesh_asset_id = hit_rec.mesh_asset_id;

        var entity_transform = entity_transforms[entity_resolved].transform;
        let hit = trace_blas(&ray, &entity_transform, mesh_asset_id);
        if (hit.normal_and_user_data.w >= 0.0) {
            let tri_id = u32(hit.normal_and_user_data.w);
            sample_rgb = primitive_debug_color(mesh_asset_id, tri_id);
            break;
        }
    }

    let prev_sum = pixel_info[pixel_index].accum_color;
    let new_count = prev_count + 1.0;
    let accum = prev_sum + vec4f(sample_rgb, 1.0);
    pixel_info[pixel_index].accum_color = accum;
    pixel_info[pixel_index].normal_and_sample_count.w = new_count;

    let inv = 1.0 / max(new_count, 1.0);
    let avg = vec4f(accum.xyz * inv, 1.0);
    textureStore(output_tex, vec2<i32>(gid.xy), avg);
}


