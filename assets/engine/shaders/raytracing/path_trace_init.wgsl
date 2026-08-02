// =============================================================================
// Path Tracer - Init Pass (Simple Monte Carlo)
// =============================================================================
// Initializes per-pixel path state for unbiased Monte Carlo path tracing.
// Generates primary rays from the active camera or reads G-buffer data.
// Resets accumulation buffer when camera moves for progressive rendering.
// =============================================================================

#include "common.wgsl"
#include "visibility/visibility_common.wgsl"

// ─────────────────────────────────────────────────────────────────────────────
// Path Tracer Parameters
// ─────────────────────────────────────────────────────────────────────────────
struct PathTracerParams {
    max_bounces: u32,          // Maximum number of light bounces
    reset_accum_flag: u32,     // 1 = camera moved, reset accumulation
    use_gbuffer: u32,          // 1 = hybrid mode (raster first hit)
    trace_rate: u32,           // 1=full res, 2=half, 4=quarter, etc.
    frame_phase: u32,          // Cycles 0 to trace_rate-1
    samples_per_pixel: u32,    // Number of samples per pixel per frame
    sample_index: u32,         // Current sample index (0 to samples_per_pixel-1)
    sampling_tile_width: u32,
    max_accumulation_frames: u32, // 0 = infinite progressive accumulation
};

// ─────────────────────────────────────────────────────────────────────────────
// Path State - Per-ray geometric and accumulation information
// ─────────────────────────────────────────────────────────────────────────────
struct PathState {
    origin_tmin: vec4<f32>,            // xyz = ray origin, w = t_min
    direction_tmax: vec4<f32>,         // xyz = ray direction, w = t_max or prim_store
    normal_section_index: vec4<f32>,   // xyz = surface normal, w = section index
    state_u32: vec4<u32>,              // x = bounce, y = alive, z = shadow_visible, w = tri_id
    hit_attr0: vec4<f32>,              // xyz = tangent (or albedo for gbuffer), w = uv.x (or roughness)
    hit_attr1: vec4<f32>,              // xyz = bitangent (or metallic,refl,emissive), w = uv.y
    shadow_origin: vec4<f32>,          // xyz = shadow ray origin, w = t_min
    shadow_direction: vec4<f32>,       // xyz = shadow ray direction, w = t_max
    shadow_radiance: vec4<f32>,        // rgb = potential light contribution, a = needs_trace
    path_weight: vec4<f32>,            // xyz = current path throughput, w = unused
    rng_sample_count: vec4<f32>,       // x = rng state, y = sample count, z = accumulated frame count
    accumulated_radiance: vec4<f32>,   // xyz = total accumulated radiance (demodulated), w = unused
    primary_albedo: vec4<f32>,         // xyz = primary hit albedo for demodulation, w = unused
};

// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
@group(1) @binding(0) var<uniform> pt_params: PathTracerParams;
@group(1) @binding(1) var<storage, read_write> path_state: array<PathState>;
@group(1) @binding(2) var visibility_entity_texture: texture_2d<u32>;
@group(1) @binding(3) var visibility_surface_texture: texture_2d<u32>;
@group(1) @binding(4) var<storage, read> entity_transforms: array<EntityTransform>;
@group(1) @binding(5) var<storage, read> meshlets: array<MeshletRecord>;
@group(1) @binding(6) var<storage, read> meshlet_vertices: array<u32>;
@group(1) @binding(7) var<storage, read> meshlet_triangles: array<u32>;
@group(1) @binding(8) var depth_texture: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(10) var gbuffer_albedo: texture_2d<f32>;
@group(1) @binding(11) var gbuffer_smra: texture_2d<f32>;
@group(1) @binding(12) var gbuffer_motion_emissive: texture_2d<f32>;
@group(1) @binding(13) var output_tex: texture_storage_2d<rgba16float, write>;

struct VisibilityFirstHit {
    valid: bool,
    world_position: vec3<f32>,
    geometric_normal: vec3<f32>,
};

fn load_visibility_first_hit(
    pixel_coord: vec2<i32>,
    resolution: vec2<u32>,
    view_index: u32
) -> VisibilityFirstHit {
    let entity_id = textureLoad(visibility_entity_texture, pixel_coord, 0).x;
    if (entity_id == INVALID_IDX) {
        return VisibilityFirstHit(false, vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let surface = textureLoad(visibility_surface_texture, pixel_coord, 0).x;
    let meshlet_index_value = unpack_surface_meshlet(surface);
    let triangle_index = unpack_surface_triangle(surface);
    let meshlet = meshlets[meshlet_index_value];

    let tri_offset = meshlet.triangle_offset + triangle_index * 3u;
    let local_index0 = meshlet_triangles[tri_offset + 0u];
    let local_index1 = meshlet_triangles[tri_offset + 1u];
    let local_index2 = meshlet_triangles[tri_offset + 2u];

    let global_index0 = meshlet_vertices[meshlet.vertex_offset + local_index0];
    let global_index1 = meshlet_vertices[meshlet.vertex_offset + local_index1];
    let global_index2 = meshlet_vertices[meshlet.vertex_offset + local_index2];

    let decoded0 = decode_vertex(vertex_buffer[global_index0]);
    let decoded1 = decode_vertex(vertex_buffer[global_index1]);
    let decoded2 = decode_vertex(vertex_buffer[global_index2]);

    let entity_transform = entity_transforms[entity_id];
    let world_position0 = entity_transform.transform * decoded0.position;
    let world_position1 = entity_transform.transform * decoded1.position;
    let world_position2 = entity_transform.transform * decoded2.position;

    let clip_pos0 = view_buffer[view_index].view_projection_matrix * world_position0;
    let clip_pos1 = view_buffer[view_index].view_projection_matrix * world_position1;
    let clip_pos2 = view_buffer[view_index].view_projection_matrix * world_position2;

    let bary = calc_full_barycentric(
        coord_to_uv(pixel_coord, resolution),
        clip_pos0,
        clip_pos1,
        clip_pos2
    );

    let world_position = interpolate_vec3(
        world_position0.xyz,
        world_position1.xyz,
        world_position2.xyz,
        bary
    );

    let geometric_normal = safe_normalize(cross(
        world_position1.xyz - world_position0.xyz,
        world_position2.xyz - world_position0.xyz
    ));

    return VisibilityFirstHit(true, world_position, geometric_normal);
}

// =============================================================================
// Main Compute Shader
// =============================================================================
@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_tex);
    
    // Compute pixel coordinates based on trace pattern
    let pixel_coords = select(
        compute_phased_pixel_coords(
            gid.x,
            res,
            pt_params.trace_rate,
            pt_params.frame_phase,
            pt_params.sampling_tile_width
        ),
        vec2<u32>(gid.x % res.x, gid.x / res.x),
        pt_params.reset_accum_flag != 0u
    );

    if (pixel_coords.x >= res.x || pixel_coords.y >= res.y) { return; }
    let pixel_index = pixel_coords.y * res.x + pixel_coords.x;

    let view_index = u32(frame_info.view_index);
    let view = view_buffer[view_index];
    
    // ─────────────────────────────────────────────────────────────────────────
    // Determine if this pixel should be traced this frame (for trace_rate > 1)
    // ─────────────────────────────────────────────────────────────────────────
    let should_trace_this_pixel = is_phased_pixel(
        pixel_coords,
        res,
        pt_params.trace_rate,
        pt_params.frame_phase,
        pt_params.sampling_tile_width
    );
    
    // ─────────────────────────────────────────────────────────────────────────
    // Reset accumulation when camera moves (only on first sample)
    // ─────────────────────────────────────────────────────────────────────────
    if (pt_params.reset_accum_flag != 0u && pt_params.sample_index == 0u) {
        let frame_id = u32(frame_info.frame_index);
        let rng_seed = hash(pixel_index ^ frame_id);
        
        path_state[pixel_index].rng_sample_count = vec4<f32>(f32(rng_seed), 0.0, 0.0, 0.0);
        path_state[pixel_index].accumulated_radiance = vec4<f32>(0.0);
    }

    // Bound the temporal history before adding this frame. Scaling the stored
    // sum and its sample count together preserves the current average while
    // giving the new frame exactly one frame of weight in the bounded history.
    if (
        should_trace_this_pixel &&
        pt_params.sample_index == 0u &&
        pt_params.max_accumulation_frames != 0u
    ) {
        let accumulated_frame_count = path_state[pixel_index].rng_sample_count.z;
        let max_frame_count = f32(pt_params.max_accumulation_frames);
        if (accumulated_frame_count >= max_frame_count) {
            let retained_frame_count = max_frame_count - 1.0;
            let history_scale = retained_frame_count / max(accumulated_frame_count, 1.0);
            path_state[pixel_index].rng_sample_count.y *= history_scale;
            path_state[pixel_index].rng_sample_count.z = retained_frame_count;
            path_state[pixel_index].accumulated_radiance *= history_scale;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Initialize path for pixels being traced this frame
    // ─────────────────────────────────────────────────────────────────────────
    if (should_trace_this_pixel && pt_params.use_gbuffer != 0u) {
        // =====================================================================
        // G-Buffer Mode: Read first hit from rasterized G-buffer
        // =====================================================================
        let pixel_coord = vec2<i32>(i32(pixel_coords.x), i32(pixel_coords.y));
        let uv = coord_to_uv(pixel_coord, res);
        let depth = textureLoad(depth_texture, pixel_coord, 0).r;
        
        let visibility_hit = load_visibility_first_hit(pixel_coord, res, view_index);
        let gbuffer_pos = reconstruct_world_position(uv, depth, view_index);
        let gbuffer_norm_data = textureLoad(gbuffer_normal, pixel_coord, 0);
        let gbuffer_norm = gbuffer_norm_data.xyz;
        let gbuffer_norm_length = length(gbuffer_norm);

        if (visibility_hit.valid || gbuffer_norm_length > 0.0) {
            // Valid geometry hit - read material properties
            let hit_pos = select(gbuffer_pos, visibility_hit.world_position, visibility_hit.valid);
            let albedo_data = textureLoad(gbuffer_albedo, pixel_coord, 0);
            let smra_data = textureLoad(gbuffer_smra, pixel_coord, 0);
            let emissive_data = textureLoad(gbuffer_motion_emissive, pixel_coord, 0).w;
            
            let ray_dir = normalize(hit_pos - view.view_position.xyz);

            var spawn_normal = select(
                safe_normalize(gbuffer_norm),
                visibility_hit.geometric_normal,
                visibility_hit.valid
            );
            if (dot(spawn_normal, ray_dir) > 0.0) {
                spawn_normal = -spawn_normal;
            }

            var shading_normal = select(
                spawn_normal,
                safe_normalize(gbuffer_norm),
                gbuffer_norm_length > 0.0
            );
            if (dot(shading_normal, ray_dir) > 0.0) {
                shading_normal = -shading_normal;
            }
            
            path_state[pixel_index].origin_tmin = vec4<f32>(hit_pos, 0.0001);
            path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, 0.0);
            path_state[pixel_index].normal_section_index = vec4<f32>(shading_normal, 0.0);
            path_state[pixel_index].hit_attr0 = vec4<f32>(albedo_data.rgb, smra_data.g); // albedo, roughness
            path_state[pixel_index].hit_attr1 = vec4<f32>(smra_data.b, smra_data.r, emissive_data, smra_data.a); // metallic, reflectance, emissive, ao
            path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0x0u); // bounce=0, alive=1, gbuffer marker
            path_state[pixel_index].primary_albedo = vec4<f32>(albedo_data.rgb, 1.0);
            path_state[pixel_index].shadow_origin = vec4<f32>(spawn_normal, 0.0);
        } else {
            // No geometry - shoot ray to evaluate sky
            let dims = vec2<f32>(f32(res.x), f32(res.y));
            let pixel_center = vec2<f32>(f32(pixel_coords.x) + 0.5, f32(pixel_coords.y) + 0.5);
            let uv = pixel_center / dims;
            let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

            let tan_half_fov = tan(0.5 * view.fov);
            let sensor_x = ndc.x * view.aspect_ratio * tan_half_fov;
            let sensor_y = ndc.y * tan_half_fov;

            let forward = normalize(view.view_direction.xyz);
            let right = normalize(view.view_right.xyz);
            let up = normalize(cross(right, forward));
            var ray_dir = normalize(forward + right * sensor_x + up * sensor_y);
            let ray_origin = view.view_position.xyz;

            path_state[pixel_index].origin_tmin = vec4<f32>(ray_origin + ray_dir * 0.001, 0.0001);
            path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, 1e30);
            path_state[pixel_index].normal_section_index = vec4<f32>(0.0);
            path_state[pixel_index].hit_attr0 = vec4<f32>(0.0);
            path_state[pixel_index].hit_attr1 = vec4<f32>(0.0);
            path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu); // miss marker
            path_state[pixel_index].primary_albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
            path_state[pixel_index].shadow_origin = vec4<f32>(0.0);
        }
    } else if (should_trace_this_pixel) {
        // =====================================================================
        // Traditional Mode: Generate primary rays from camera
        // =====================================================================
        let dims = vec2<f32>(f32(res.x), f32(res.y));
        let pixel_center = vec2<f32>(f32(pixel_coords.x) + 0.5, f32(pixel_coords.y) + 0.5);
        let uv = pixel_center / dims;
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

        let tan_half_fov = tan(0.5 * view.fov);
        let sensor_x = ndc.x * view.aspect_ratio * tan_half_fov;
        let sensor_y = ndc.y * tan_half_fov;

        let forward = normalize(view.view_direction.xyz);
        let right = normalize(view.view_right.xyz);
        let up = normalize(cross(right, forward));
        var ray_dir = normalize(forward + right * sensor_x + up * sensor_y);
        let ray_origin = view.view_position.xyz;

        path_state[pixel_index].origin_tmin = vec4<f32>(ray_origin + ray_dir * 0.001, 0.0001);
        path_state[pixel_index].direction_tmax = vec4<f32>(ray_dir, 1e30);
        path_state[pixel_index].normal_section_index = vec4<f32>(0.0);
        path_state[pixel_index].hit_attr0 = vec4<f32>(0.0);
        path_state[pixel_index].hit_attr1 = vec4<f32>(0.0);
        path_state[pixel_index].state_u32 = vec4<u32>(0u, 1u, 0u, 0xffffffffu);
        path_state[pixel_index].primary_albedo = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        path_state[pixel_index].shadow_origin = vec4<f32>(0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Initialize fresh path state for this sample
    // ─────────────────────────────────────────────────────────────────────────
    if (should_trace_this_pixel || pt_params.reset_accum_flag != 0u) {
        path_state[pixel_index].shadow_direction = vec4<f32>(0.0);
        path_state[pixel_index].shadow_radiance = vec4<f32>(0.0);
        path_state[pixel_index].path_weight = vec4<f32>(1.0, 1.0, 1.0, 0.0);
        
        // Advance RNG for this sample
        var rng = u32(path_state[pixel_index].rng_sample_count.x);
        if (rng == 0u) {
            let frame_id = u32(frame_info.frame_index);
            rng = hash(pixel_index ^ frame_id ^ pt_params.sample_index);
        } else {
            rng = random_seed(rng);
        }
        path_state[pixel_index].rng_sample_count.x = f32(rng);
    }
}
