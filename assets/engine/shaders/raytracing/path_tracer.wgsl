#include "common.wgsl"
#include "acceleration_common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    unused: u32,
};

struct PixelInfo {
    position_and_rng: vec4<f32>,
    normal_and_sample_count: vec4<f32>,
    accum_color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> pixel_info: array<PixelInfo>;
@group(1) @binding(2) var<storage, read> tlas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(3) var<storage, read> blas_bvh4_nodes: array<BVH4Node>;
@group(1) @binding(4) var<storage, read> blas_directory: array<u32>;
@group(1) @binding(5) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(6) var<storage, read> mesh_asset_ids: array<u32>;
@group(1) @binding(7) var position_tex: texture_2d<f32>;
@group(1) @binding(8) var normal_tex: texture_2d<f32>;
@group(1) @binding(9) var output_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(position_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = gid.y * res.x + gid.x;

    // Reset path (clear hit) if requested
    if (pt_params.reset_accum_flag != 0u) {
        let rng = hash(pixel_index ^ u32(frame_info.frame_index));
        pixel_info[pixel_index].position_and_rng = vec4f(0.0, 0.0, 0.0, f32(rng)); // pos, valid=0
        pixel_info[pixel_index].normal_and_sample_count = vec4f(0.0, 0.0, 0.0, 0.0); // normal
        pixel_info[pixel_index].accum_color = vec4f(0.0);
        textureStore(output_tex, vec2<i32>(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Bootstrap primary ray from GBuffer position/normal (will be replaced by TLAS traversal)
    let pos = textureLoad(position_tex, vec2<i32>(gid.xy), 0).xyz;
    let nrm = textureLoad(normal_tex, vec2<i32>(gid.xy), 0).xyz;
    let valid = !isinf(pos.x) && !isinf(pos.y) && !isinf(pos.z);

    var rng = u32(pixel_info[pixel_index].position_and_rng.w);
    pixel_info[pixel_index].position_and_rng = vec4f(pos, f32(rng));
    pixel_info[pixel_index].normal_and_sample_count = vec4f(normalize(nrm), 0.0);

    // Advance RNG deterministically for next stages
    if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
    rng = random_seed(rng);
    pixel_info[pixel_index].position_and_rng.w = f32(rng);

    var sample_rgb = vec3f(0.0);
    if (valid) {
        let n = normalize(nrm) * 0.5 + 0.5;
        sample_rgb = n;
    }

    // Accumulate
    let prev_sum = pixel_info[pixel_index].accum_color;
    let prev_count = pixel_info[pixel_index].normal_and_sample_count.w;
    let new_count = prev_count + 1.0;
    let accum = prev_sum + vec4f(sample_rgb, 1.0);
    pixel_info[pixel_index].accum_color = accum;
    pixel_info[pixel_index].normal_and_sample_count.w = new_count;

    let inv = 1.0 / max(new_count, 1.0);
    let avg = vec4f(accum.xyz * inv, 1.0);
    textureStore(output_tex, vec2<i32>(gid.xy), avg);
}


