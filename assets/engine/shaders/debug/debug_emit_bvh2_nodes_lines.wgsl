#include "common.wgsl"
#include "acceleration_common.wgsl"

// ========================================================================================
// Constants
// ========================================================================================

const LINES_PER_BOX = 12u;
const BOUNDS_COLOR = vec4f(0.0, 1.0, 0.0, 1.0);

const EDGES: array<vec2<u32>, 12> = array<vec2<u32>, 12>(
    vec2u(0, 1), vec2u(1, 3), vec2u(3, 2), vec2u(2, 0),
    vec2u(4, 5), vec2u(5, 7), vec2u(7, 6), vec2u(6, 4),
    vec2u(0, 4), vec2u(1, 5), vec2u(2, 6), vec2u(3, 7),
);

// ========================================================================================
// Data Structures
// ========================================================================================

struct LineData { 
    color_and_width: vec4f,
};

// ========================================================================================
// Helper Functions
// ========================================================================================

fn create_line_transform(start: vec3f, end: vec3f) -> mat4x4f {
    let dir = end - start;
    let len = length(dir);
    let is_short = len < 0.0001;
    let safe_len = select(len, 1.0, is_short);

    let n = dir / safe_len;
    var up = vec3f(0.0, 1.0, 0.0);
    let near_up = abs(dot(n, up)) > 0.99;
    up = select(up, vec3f(0.0, 0.0, 1.0), near_up);

    let right = normalize(cross(n, up));
    let true_up = normalize(cross(right, n));

    let rot = mat4x4f(
        vec4f(n, 0.0),
        vec4f(true_up, 0.0),
        vec4f(right, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0),
    );

    let trans = mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(start, 1.0),
    );

    let scale = mat4_from_scaling(vec3f(len, 1.0, 1.0));
    return trans * rot * scale;
}

fn corner(min_p: vec3f, max_p: vec3f, idx: u32) -> vec3f {
    let x_sel = select(min_p.x, max_p.x, (idx & 1u) != 0u);
    let y_sel = select(min_p.y, max_p.y, (idx & 2u) != 0u);
    let z_sel = select(min_p.z, max_p.z, (idx & 4u) != 0u);
    return vec3f(x_sel, y_sel, z_sel);
}

// ========================================================================================
// Buffers 
// ========================================================================================

@group(1) @binding(0) var<storage, read_write> out_transforms: array<mat4x4f>;
@group(1) @binding(1) var<storage, read_write> out_line_data: array<LineData>;
@group(1) @binding(2) var<storage, read> aabb_bounds: array<AABB>;

// ========================================================================================
// Main
// ========================================================================================

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3u) {
    let node = gid.x;
    if (node >= arrayLength(&aabb_bounds)) { return; }

    let b = aabb_bounds[node];
    let min_p = b.min.xyz;
    let max_p = b.max.xyz;
    let valid = !is_leaf(b);

    let has_volume = all(max_p > min_p);
    let is_active = valid && has_volume;
    let width = select(0.04, 0.0, !is_active);

    for (var e: u32 = 0u; e < LINES_PER_BOX; e = e + 1u) {
        let base = node * LINES_PER_BOX + e;

        let idx0 = EDGES[e].x;
        let idx1 = EDGES[e].y;
        let p0 = corner(min_p, max_p, idx0);
        let p1 = corner(min_p, max_p, idx1);

        let transform = create_line_transform(p0, p1);
        out_transforms[base][0] = select(transform[0], identity_matrix[0], !is_active);
        out_transforms[base][1] = select(transform[1], identity_matrix[1], !is_active);
        out_transforms[base][2] = select(transform[2], identity_matrix[2], !is_active);
        out_transforms[base][3] = select(transform[3], identity_matrix[3], !is_active);

        out_line_data[base].color_and_width = vec4f(BOUNDS_COLOR.rgb, width);
    }
}



