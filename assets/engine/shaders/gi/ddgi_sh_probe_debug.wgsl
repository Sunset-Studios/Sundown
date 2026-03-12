// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║              DDGI SPHERICAL HARMONICS PROBE DEBUG VISUALIZATION           ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Renders probe spheres with SH-reconstructed radiance for debugging.      ║
// ║  Visualizes the SH representation quality by evaluating radiance at       ║
// ║  each point on the probe sphere surface.                                  ║
// ║                                                                           ║
// ║  Key features:                                                            ║
// ║  • Ray-sphere intersection for freestanding probe rendering               ║
// ║  • SH radiance evaluation at sphere surface points                        ║
// ║  • Depth-based occlusion against scene geometry                           ║
// ║  • AABB culling for efficient probe grid traversal                        ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

#include "common.wgsl"
#include "postprocess_common.wgsl"
#include "gi/ddgi_common.wgsl"

// =============================================================================
// BINDINGS
// =============================================================================

@group(1) @binding(0) var<uniform> ddgi_params: DDGIParams;
@group(1) @binding(1) var<storage, read_write> sh_probes: array<u32>;
@group(1) @binding(2) var<storage, read_write> probe_states: array<ProbeStateData>;
@group(1) @binding(3) var scene_color: texture_2d<f32>;
@group(1) @binding(4) var depth_texture: texture_2d<f32>;
@group(1) @binding(5) var output_debug: texture_storage_2d<rgba16float, write>;

const STATE_DEBUG_SHOW_ALL: bool = false;
const STATE_DEBUG_COLOR_OVERLAY_STRENGTH: f32 = 0.0; // Tweak this to show debug colors for probe states (0.0 = no overlay, 1.0 = full overlay)

// =============================================================================
// PROBE STATE DEBUG COLORS
// =============================================================================
// Visual color key for probe states (used as an overlay on the probe radiance):
// - UNINITIALIZED     : gray
// - OFF (in wall)     : dark red
// - SLEEPING (empty)  : blue
// - NEWLY_AWAKE       : orange
// - NEWLY_VIGILANT    : yellow
// - VIGILANT          : green
// - AWAKE             : cyan
// - unknown           : magenta
fn ddgi_probe_state_debug_color(state: u32) -> vec3<f32> {
    var color = vec3<f32>(1.0, 0.0, 1.0);
    color = select(color, vec3<f32>(0.65, 0.65, 0.65), state == PROBE_STATE_UNINITIALIZED);
    color = select(color, vec3<f32>(0.35, 0.05, 0.05), state == PROBE_STATE_OFF);
    color = select(color, vec3<f32>(0.10, 0.20, 0.85), state == PROBE_STATE_SLEEPING);
    color = select(color, vec3<f32>(1.00, 0.45, 0.05), state == PROBE_STATE_NEWLY_AWAKE);
    color = select(color, vec3<f32>(1.00, 0.95, 0.05), state == PROBE_STATE_NEWLY_VIGILANT);
    color = select(color, vec3<f32>(0.10, 0.90, 0.10), state == PROBE_STATE_VIGILANT);
    color = select(color, vec3<f32>(0.10, 0.95, 0.95), state == PROBE_STATE_AWAKE);
    return color;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruct world-space ray direction from pixel UV
// ─────────────────────────────────────────────────────────────────────────────
fn sh_debug_world_ray_direction(pixel_uv: vec2<f32>, view_index: u32) -> vec3<f32> {
    // WebGPU NDC: x,y in [-1,1], z in [0,1]
    let ndc = vec2<f32>(pixel_uv.x * 2.0 - 1.0, (1.0 - pixel_uv.y) * 2.0 - 1.0);
    let inv_vp = view_buffer[view_index].inverse_view_projection_matrix;
    
    let far_h = inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let far_ws = far_h.xyz / max(far_h.w, 1e-8);
    
    let ro = view_buffer[view_index].view_position.xyz;
    return safe_normalize(far_ws - ro);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray-sphere intersection
// Returns nearest positive t, or -1.0 if no hit
// ─────────────────────────────────────────────────────────────────────────────
fn sh_debug_ray_sphere_intersect(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    let t = -b - sqrt(disc);
    return select(-1.0, t, t > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray-AABB intersection
// Returns (t_enter, t_exit). If miss, t_enter > t_exit
// ─────────────────────────────────────────────────────────────────────────────
fn sh_debug_ray_aabb_intersect(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv_rd = vec3<f32>(
        1.0 / max(abs(rd.x), 1e-8) * select(1.0, -1.0, rd.x < 0.0),
        1.0 / max(abs(rd.y), 1e-8) * select(1.0, -1.0, rd.y < 0.0),
        1.0 / max(abs(rd.z), 1e-8) * select(1.0, -1.0, rd.z < 0.0)
    );
    let t0 = (bmin - ro) * inv_rd;
    let t1 = (bmax - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_enter, t_exit);
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruct world position from depth
// ─────────────────────────────────────────────────────────────────────────────
fn sh_debug_world_position_from_depth(pixel_uv: vec2<f32>, depth: f32, view_index: u32) -> vec3<f32> {
    let ndc = vec2<f32>(pixel_uv.x * 2.0 - 1.0, (1.0 - pixel_uv.y) * 2.0 - 1.0);
    let inv_vp = view_buffer[view_index].inverse_view_projection_matrix;
    let p_h = inv_vp * vec4<f32>(ndc, depth, 1.0);
    return p_h.xyz / max(p_h.w, 1e-8);
}

// =============================================================================
// CASCADE PROBE TRAVERSAL
// =============================================================================
// Traverses a single cascade's probe grid using 3D DDA and returns the nearest
// probe hit. Returns vec2(hit_t, probe_index) or vec2(1e30, 0) if no hit.
// Uses clipmap-style selection to only show probes in each cascade's "shell".

fn traverse_cascade_probes(
    cascade_index: u32,
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    probe_radius: f32,
    current_best_t: f32
) -> vec2<f32> {
    let spacing = ddgi_cascade_spacing(&ddgi_params, cascade_index);
    let origin = ddgi_cascade_origin(&ddgi_params, cascade_index);
    
    let dims = vec3<u32>(
        u32(ddgi_params.probe_grid_dims.x),
        u32(ddgi_params.probe_grid_dims.y),
        u32(ddgi_params.probe_grid_dims.z)
    );
    let max_vertex = vec3<f32>(
        f32(max(dims.x, 1u) - 1u),
        f32(max(dims.y, 1u) - 1u),
        f32(max(dims.z, 1u) - 1u)
    );
    let bmin_ws = origin;
    let bmax_ws = origin + max_vertex * spacing;
    
    // Expand AABB by probe radius
    let t_range = sh_debug_ray_aabb_intersect(
        ray_origin,
        ray_direction,
        bmin_ws,
        bmax_ws
    );
    
    // Early exit if ray misses cascade AABB or enters after current best
    if (t_range.x > t_range.y || t_range.x > current_best_t) {
        return vec2<f32>(1e30, 0.0);
    }
    
    // Grid traversal setup (3D DDA)
    let cell_dims = vec3<u32>(
        select(0u, dims.x - 1u, dims.x > 1u),
        select(0u, dims.y - 1u, dims.y > 1u),
        select(0u, dims.z - 1u, dims.z > 1u)
    );
    
    let ro_vs = (ray_origin - origin) / max(spacing, 1e-8);
    let rd_vs = ray_direction / max(spacing, 1e-8);
    let vs_range = sh_debug_ray_aabb_intersect(
        ro_vs,
        rd_vs,
        vec3<f32>(0.0),
        vec3<f32>(f32(cell_dims.x), f32(cell_dims.y), f32(cell_dims.z))
    );
    
    var t_enter = max(vs_range.x, 0.0);
    let t_exit = vs_range.y;
    
    if (t_enter > t_exit) {
        return vec2<f32>(1e30, 0.0);
    }
    
    // Starting cell
    let p0 = ro_vs + rd_vs * (t_enter + 1e-4);
    var cell = vec3<i32>(
        clamp(i32(floor(p0.x)), 0, i32(max(cell_dims.x, 1u) - 1u)),
        clamp(i32(floor(p0.y)), 0, i32(max(cell_dims.y, 1u) - 1u)),
        clamp(i32(floor(p0.z)), 0, i32(max(cell_dims.z, 1u) - 1u))
    );
    
    let step = vec3<i32>(
        select(-1, 1, rd_vs.x >= 0.0),
        select(-1, 1, rd_vs.y >= 0.0),
        select(-1, 1, rd_vs.z >= 0.0)
    );
    
    let next_boundary = vec3<f32>(
        f32(select(cell.x, cell.x + 1, step.x > 0)),
        f32(select(cell.y, cell.y + 1, step.y > 0)),
        f32(select(cell.z, cell.z + 1, step.z > 0))
    );
    
    var t_max = vec3<f32>(
        select(1e30, (next_boundary.x - ro_vs.x) / rd_vs.x, abs(rd_vs.x) > 1e-8),
        select(1e30, (next_boundary.y - ro_vs.y) / rd_vs.y, abs(rd_vs.y) > 1e-8),
        select(1e30, (next_boundary.z - ro_vs.z) / rd_vs.z, abs(rd_vs.z) > 1e-8)
    );
    
    let t_delta = vec3<f32>(
        select(1e30, f32(step.x) / rd_vs.x, abs(rd_vs.x) > 1e-8),
        select(1e30, f32(step.y) / rd_vs.y, abs(rd_vs.y) > 1e-8),
        select(1e30, f32(step.z) / rd_vs.z, abs(rd_vs.z) > 1e-8)
    );
    
    // Traverse grid and find nearest probe sphere hit
    var hit_t = current_best_t;
    var hit_probe_index = 0u;
    var hit = false;
    
    let max_steps = 32u;
    for (var iter = 0u; iter < max_steps; iter = iter + 1u) {
        // Test 8 vertices of current cell
        let base = vec3<u32>(u32(cell.x), u32(cell.y), u32(cell.z));
        
        for (var oz = 0u; oz < 2u; oz = oz + 1u) {
            for (var oy = 0u; oy < 2u; oy = oy + 1u) {
                for (var ox = 0u; ox < 2u; ox = ox + 1u) {
                    let v = base + vec3<u32>(ox, oy, oz);
                    let probe_idx = ddgi_probe_index_from_coord(&ddgi_params, cascade_index, v);
                    let state = probe_state_get_state(probe_states[probe_idx].packed_state);

                    if (!STATE_DEBUG_SHOW_ALL && !probe_state_is_active(state)) {
                        continue;
                    }

                    // Clipmap selection: only show probes in this cascade's "shell"
                    if (!ddgi_probe_in_cascade_shell(&ddgi_params, probe_idx)) {
                        continue;
                    }

                    let center = ddgi_probe_world_position_from_coord(
                        &ddgi_params,
                        cascade_index,
                        v
                    );
                    
                    let t = sh_debug_ray_sphere_intersect(ray_origin, ray_direction, center, probe_radius * f32(cascade_index + 1u));
                    let valid = t > 0.0 && t >= t_range.x && t <= t_range.y && t < hit_t;
                    hit_t = select(hit_t, t, valid);
                    hit_probe_index = select(hit_probe_index, probe_idx, valid);
                    hit = hit || valid;
                }
            }
        }
        
        if (hit) {
            break;
        }
        
        // Advance to next cell
        let step_x = t_max.x <= t_max.y && t_max.x <= t_max.z;
        let step_y = (!step_x) && (t_max.y <= t_max.z);
        let step_z = (!step_x) && (!step_y);
        
        cell.x = cell.x + select(0, step.x, step_x);
        cell.y = cell.y + select(0, step.y, step_y);
        cell.z = cell.z + select(0, step.z, step_z);
        
        t_enter = select(t_enter, t_max.x, step_x);
        t_enter = select(t_enter, t_max.y, step_y);
        t_enter = select(t_enter, t_max.z, step_z);
        
        t_max.x = select(t_max.x, t_max.x + t_delta.x, step_x);
        t_max.y = select(t_max.y, t_max.y + t_delta.y, step_y);
        t_max.z = select(t_max.z, t_max.z + t_delta.z, step_z);
        
        let out_of_bounds =
            cell.x < 0 || cell.y < 0 || cell.z < 0 ||
            cell.x >= i32(cell_dims.x) || cell.y >= i32(cell_dims.y) || cell.z >= i32(cell_dims.z) ||
            t_enter > t_exit;
        
        if (out_of_bounds) {
            break;
        }
    }
    
    return select(vec2<f32>(1e30, 0.0), vec2<f32>(hit_t, f32(hit_probe_index)), hit);
}

// =============================================================================
// MAIN COMPUTE SHADER
// =============================================================================

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = textureDimensions(output_debug);
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }
    
    let pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let scene = textureLoad(scene_color, pixel_coord, 0).rgb;
    
    let probe_radius = ddgi_params.probe_grid_dims.w;
    let cascade_count = ddgi_cascade_count(&ddgi_params);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Construct camera ray for this pixel
    // ─────────────────────────────────────────────────────────────────────────
    let view_index = u32(frame_info.view_index);
    let ray_origin = view_buffer[view_index].view_position.xyz;
    let pixel_uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(f32(res.x), f32(res.y));
    let ray_direction = sh_debug_world_ray_direction(pixel_uv, view_index);
    
    // ─────────────────────────────────────────────────────────────────────────
    // Traverse all cascades and find nearest probe hit
    // ─────────────────────────────────────────────────────────────────────────
    var hit_t = 1e30;
    var hit_probe_index = 0u;
    var hit = false;
    
    for (var c = 0u; c < cascade_count; c = c + 1u) {
        let result = traverse_cascade_probes(c, ray_origin, ray_direction, probe_radius, hit_t);
        if (result.x < hit_t) {
            hit_t = result.x;
            hit_probe_index = u32(result.y);
            hit = true;
        }
    }
    
    if (!hit) {
        textureStore(output_debug, pixel_coord, vec4<f32>(scene, 1.0));
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Depth-based occlusion check
    // ─────────────────────────────────────────────────────────────────────────
    let depth = textureSampleLevel(depth_texture, non_filtering_sampler, pixel_uv, 0.0).r;
    if (depth < 1.0) {
        let p_surface = sh_debug_world_position_from_depth(pixel_uv, depth, view_index);
        let t_surface = dot(p_surface - ray_origin, ray_direction);
        let occluded = t_surface > 0.0 && hit_t > (t_surface + 1e-3);
        if (occluded) {
            textureStore(output_debug, pixel_coord, vec4<f32>(scene, 1.0));
            return;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sample SH probe and evaluate irradiance at sphere surface point
    // - We visualize a Lambertian-equivalent radiance preview: L = E / PI
    // - This tends to match how the probe field contributes to diffuse surfaces
    // ─────────────────────────────────────────────────────────────────────────
    let probe_center = ddgi_probe_world_position_from_index(&ddgi_params, hit_probe_index);
    let hit_pos = ray_origin + ray_direction * hit_t;
    let sphere_normal = safe_normalize(hit_pos - probe_center);
    
    // Read SH probe and evaluate diffuse irradiance for the sphere surface normal.
    let probe_sh = ddgi_sh_probe_read(&sh_probes, hit_probe_index);
    let sphere_irradiance = ddgi_sh_evaluate_irradiance(probe_sh, sphere_normal) * ddgi_params.indirect_boost;

    // Lambertian-equivalent radiance preview (albedo = 1): L = E / PI
    let sphere_radiance = max(sphere_irradiance, vec3<f32>(0.0)) * (1.0 / PI);
    
    // Tone map + display gamma for debug readability.
    let exposure = 1.2;
    let mapped_color = aces_tonemapping(sphere_radiance, exposure);
    let display_color = pow(clamp(mapped_color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));

    // -------------------------------------------------------------------------
    // State overlay (high-contrast state visualization)
    // -------------------------------------------------------------------------
    let hit_state = probe_state_get_state(probe_states[hit_probe_index].packed_state);
    let state_color = ddgi_probe_state_debug_color(hit_state);

    // Blend: keep some radiance info but strongly tint by state.
    let final_color = mix(display_color, state_color, STATE_DEBUG_COLOR_OVERLAY_STRENGTH);

    textureStore(output_debug, pixel_coord, vec4<f32>(final_color, 1.0));
}
