// =============================================================================
// Path Tracer - Shade Pass
// - Consumes TLAS hits and performs BLAS traversal
// - Shades the hit (debug color) and accumulates
// - Spawns the next ray in `path_state` for next bounce
// =============================================================================
#include "common.wgsl"
#include "acceleration_common.wgsl"
#include "blas_common.wgsl"
#include "lighting_common.wgsl"

struct PathTracerParams {
    max_bounces: u32,
    spp_per_frame: u32,
    reset_accum_flag: u32,
    max_spp: u32,
};

struct PathState {
    origin_tmin: vec4<f32>,
    direction_tmax: vec4<f32>,
    normal: vec4<f32>,
    throughput: vec4<f32>,
    state_u32: vec4<u32>, // x=bounce, y=alive(0/1)
    hit_attr0: vec4<f32>, // xyz = world_tangent, w = uv.x
    hit_attr1: vec4<f32>, // xyz = world_bitangent, w = uv.y
    padding: vec4<f32>,
};

struct PixelHitInfo {
    rng: f32,
    sample_count: f32,
    prim_id: f32,
    frame_stamp: f32,
    accum_color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var<storage, read_write> pixel_info: array<PixelHitInfo>;
@group(1) @binding(3) var<storage, read> blas_atlas: BLASAtlas;
@group(1) @binding(4) var<storage, read> material_params: array<StandardMaterialParams>;
@group(1) @binding(5) var<storage, read> material_table_offset: array<u32>;
@group(1) @binding(6) var<storage, read> material_palette: array<u32>;
@group(1) @binding(7) var<storage, read> index_buffer: array<u32>;
@group(1) @binding(8) var texture_pool_albedo: texture_2d_array<f32>;
@group(1) @binding(9) var texture_pool_normal: texture_2d_array<f32>;
@group(1) @binding(10) var texture_pool_roughness: texture_2d_array<f32>;
@group(1) @binding(11) var texture_pool_metallic: texture_2d_array<f32>;
@group(1) @binding(12) var texture_pool_ao: texture_2d_array<f32>;
@group(1) @binding(13) var texture_pool_height: texture_2d_array<f32>;
@group(1) @binding(14) var texture_pool_specular: texture_2d_array<f32>;
@group(1) @binding(15) var texture_pool_emission: texture_2d_array<f32>;
@group(1) @binding(16) var output_tex: texture_storage_2d<rgba16float, write>;

fn sample_texture_or_vec4_param_handle(
    tex_handle: u32,
    uv_coords: vec2<precision_float>,
    param_val: vec4<precision_float>,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> vec4<precision_float> {
    if ((flag & 1u) != 0u) {
        return sample_handle_rgba(tex_handle, uv_coords, pool, lod);
    }
    return param_val;
}

fn sample_texture_or_float_param_handle(
    tex_handle: u32,
    uv_coords: vec2<precision_float>,
    param_val: precision_float,
    flag: u32,
    pool: texture_2d_array<f32>,
    lod: f32
) -> precision_float {
    if ((flag & 1u) != 0u) {
        let sampled_val = sample_handle_rgba(tex_handle, uv_coords, pool, lod);
        let channel_index = (flag >> 1u) & 3u;
        return select(select(select(sampled_val.r, sampled_val.g, channel_index == 1u), sampled_val.b, channel_index == 2u), sampled_val.a, channel_index == 3u);
    }
    return param_val;
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let pixel_index = gid.y * res.x + gid.x;

    var info = pixel_info[pixel_index];
    if (pt_params.max_spp != 0u && info.sample_count >= f32(pt_params.max_spp)) {
        let denom = max(info.sample_count, 1.0);
        let avg = vec4f(info.accum_color.xyz / denom, 1.0);
        textureStore(output_tex, vec2<i32>(gid.xy), avg);
        return;
    }

    let ps = path_state[pixel_index];
    var sample_rgb = vec3f(0.0);

    if (ps.state_u32.w != 0xffffffffu) {
        let tri_id = ps.state_u32.w;
        let mesh_id = ps.state_u32.z;

        // Lookup entity row (stored in direction_tmax.w as f32) -> resolve to entity index
        let prim_store = u32(ps.direction_tmax.w);
        let entity_palette_base = material_table_offset[prim_store];

        // Derive section_index by reading any triangle vertex's section from vertex_buffer
        let mesh_entry = atlas_load_directory_entry(mesh_id);
        let first_vertex = mesh_entry.first_vertex;
        let first_index = mesh_entry.first_index;
        let i0 = index_buffer[first_index + tri_id * 3u + 0u];
        let section_index = u32(vertex_buffer[first_vertex + i0].section_index);

        // Resolve material
        let mat_params_index = material_palette[entity_palette_base + section_index];
        let material = material_params[mat_params_index];

        // Use interpolated UVs produced by hit stage
        let tiling = material.emission_roughness_metallic_tiling.w;
        let base_uv = vec2f(ps.hit_attr0.w, ps.hit_attr1.w) * tiling;

        let albedo = sample_texture_or_vec4_param_handle(
            u32(material.albedo_handle),
            base_uv,
            material.albedo,
            u32(material.texture_flags1.x),
            texture_pool_albedo,
            0.0
        ).xyz;
        let roughness = sample_texture_or_float_param_handle(
            u32(material.roughness_handle),
            base_uv,
            material.emission_roughness_metallic_tiling.y,
            u32(material.texture_flags1.z),
            texture_pool_roughness,
            0.0
        );
        let metallic = sample_texture_or_float_param_handle(
            u32(material.metallic_handle),
            base_uv,
            material.emission_roughness_metallic_tiling.z,
            u32(material.texture_flags1.w),
            texture_pool_metallic,
            0.0
        );

        // Build world-space TBN and derive normal from normal map if enabled
        let world_t = normalize(ps.hit_attr0.xyz);
        let world_b = normalize(ps.hit_attr1.xyz);
        let world_n = normalize(ps.normal.xyz);
        var n = world_n;
        if ((u32(material.texture_flags1.y) & 1u) != 0u) {
            let tbn = mat3x3<precision_float>(world_t, world_b, world_n);
            let nm = sample_handle_rgba(
                u32(material.normal_handle),
                base_uv,
                texture_pool_normal,
                0.0
            ).xyz * 2.0 - 1.0;
            n = normalize(tbn * nm);
        }

        // PBR BRDF evaluation for irradiance accumulation
        let hit_pos = ps.origin_tmin.xyz;
        // The incoming ray direction was stored in direction_tmax.xyz by the hit pass
        let v_dir = -normalize(ps.direction_tmax.xyz);

        let current_throughput = ps.throughput.xyz;

        // Sample a cosine-weighted direction for bounce and evaluate BRDF
        var rng = u32(info.rng);
        if (rng == 0u) { rng = hash(pixel_index ^ u32(frame_info.frame_index)); }
        else { rng = random_seed(rng); }
        let r1 = rand_float(rng);
        rng = random_seed(rng);
        let r2 = rand_float(rng);
        info.rng = f32(rng);

        let phi = 2.0 * 3.14159265359 * r1;
        let cos_theta = sqrt(1.0 - r2);
        let sin_theta = sqrt(r2);

        // Choose a robust up vector: default z-up, fallback to x-axis when nearly parallel
        let up = select(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), abs(n.z) > 0.999);
        let tangent = normalize(cross(up, n));
        let bitangent = normalize(cross(n, tangent));
        let dir_local = vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
        let l = normalize(tangent * dir_local.x + bitangent * dir_local.y + n * dir_local.z);

        // Evaluate BRDF for RT step (no direct light struct)
        let reflectance = 0.5; // dielectric sqrt(f0/0.16)
        let clear_coat = 0.0;
        let clear_coat_roughness = 0.5;
        let brdf_result = calculate_brdf_rt(
            n,
            v_dir,
            l,
            albedo,
            roughness,
            metallic,
            reflectance,
            clear_coat,
            clear_coat_roughness
        );
        // For cosine-weighted sampling: BRDF already includes n_dot_l, PDF = n_dot_l/PI
        // So contribution = BRDF * n_dot_l / PDF = BRDF * PI
        sample_rgb = brdf_result * PI * current_throughput;

        let alive_next = select(0u, 1u, (ps.state_u32.x + 1u) < pt_params.max_bounces);
        // Spawn next ray from hit position along sampled direction
        // Offset along surface normal to avoid self-intersection
        path_state[pixel_index].origin_tmin = vec4f(hit_pos + n * 0.001, 0.0001);
        // Store direction for next bounce and reset tmax
        path_state[pixel_index].direction_tmax = vec4f(l, 1e30);
        // Restore throughput for next bounce (will be overwritten by hit pass with ray direction)
        path_state[pixel_index].throughput = vec4f(current_throughput, 0.0);
        // Store bounce count
        path_state[pixel_index].state_u32.x = ps.state_u32.x + 1u;
        // Store alive flag
        path_state[pixel_index].state_u32.y = alive_next;
        // Clear mesh id for next stage
        path_state[pixel_index].state_u32.z = 0u;
        // Clear triangle id for next stage
        path_state[pixel_index].state_u32.w = 0xffffffffu;

        info.sample_count = info.sample_count + 1.0;
    }

    // Accumulate contribution from this bounce
    let accum = info.accum_color + vec4f(sample_rgb, 1.0);
    info.accum_color = accum;
    pixel_info[pixel_index] = info;
    let denom = max(info.sample_count, 1.0);
    let avg = vec4f(accum.xyz / denom, 1.0);
    textureStore(output_tex, vec2<i32>(gid.xy), avg);
}


