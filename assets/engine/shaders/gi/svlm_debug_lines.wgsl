#include "gi/svlm_common.wgsl"

// Brick debug line generation.
//
// This pass is run on demand from JS when the bake changes or the debug level
// filter changes. It expands each visible leaf brick into twelve line records
// consumed by the existing line renderer.

struct LineData {
    color_and_width: vec4<f32>,
    transform: mat4x4<f32>,
};

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<storage, read> leaf_bricks: array<SVLMLeafBrick>;
@group(1) @binding(3) var<storage, read_write> line_data: array<LineData>;

const SVLM_EDGE_START = array<u32, 12>(
    0u, 1u, 3u, 2u,
    4u, 5u, 7u, 6u,
    0u, 1u, 2u, 3u
);

const SVLM_EDGE_END = array<u32, 12>(
    1u, 3u, 2u, 0u,
    5u, 7u, 6u, 4u,
    4u, 5u, 6u, 7u
);

fn svlm_level_color(level: u32) -> vec3<f32> {
    let index = level % 6u;
    if (index == 0u) { return vec3<f32>(0.20, 0.95, 0.72); } // Mint Green
    if (index == 1u) { return vec3<f32>(0.38, 0.68, 1.00); } // Sky Blue
    if (index == 2u) { return vec3<f32>(1.00, 0.77, 0.25); } // Gold/Yellow
    if (index == 3u) { return vec3<f32>(1.00, 0.38, 0.46); } // Coral/Red
    if (index == 4u) { return vec3<f32>(0.72, 0.54, 1.00); } // Lavender/Purple
    return vec3<f32>(0.65, 1.00, 0.32);
}

fn svlm_box_corner(origin: vec3<f32>, size: f32, index: u32) -> vec3<f32> {
    return origin + vec3<f32>(
        select(0.0, size, (index & 1u) != 0u),
        select(0.0, size, (index & 2u) != 0u),
        select(0.0, size, (index & 4u) != 0u)
    );
}

fn svlm_clear_line(line_index: u32) {
    line_data[line_index].color_and_width = vec4<f32>(0.0);
    line_data[line_index].transform = mat4x4<f32>(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
}

fn svlm_write_line(line_index: u32, start: vec3<f32>, end: vec3<f32>, color: vec3<f32>, width: f32) {
    let dir = end - start;
    line_data[line_index].color_and_width = vec4<f32>(color, width);
    line_data[line_index].transform = mat4x4<f32>(
        vec4<f32>(dir, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(start, 1.0)
    );
}

@compute @workgroup_size(128, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_slot = gid.x;
    let leaf_count = min(atomicLoad(&svlm_counters.leaf_count), arrayLength(&leaf_bricks));
    if (leaf_slot >= leaf_count) {
        return;
    }

    let line_base = leaf_slot * 12u;
    let leaf = leaf_bricks[leaf_slot];
    let level = leaf.level;
    let debug_level = i32(svlm_params.debug_level);
    let filter_enabled = debug_level >= 0;
    let selected_level = u32(debug_level);
    if (filter_enabled && level != selected_level) {
        // svlm debug <level> uses the same parameter for bricks and probes so
        // both debug views can isolate one refinement level at a time.
        for (var edge = 0u; edge < 12u; edge = edge + 1u) {
            svlm_clear_line(line_base + edge);
        }
        return;
    }

    let origin = vec3<f32>(
        leaf.origin_x,
        leaf.origin_y,
        leaf.origin_z
    );
    let size = leaf.size;
    let color = svlm_level_color(level);
    let width = max(0.015, size * 0.006);

    for (var edge = 0u; edge < 12u; edge = edge + 1u) {
        let start = svlm_box_corner(origin, size, SVLM_EDGE_START[edge]);
        let end = svlm_box_corner(origin, size, SVLM_EDGE_END[edge]);
        svlm_write_line(line_base + edge, start, end, color, width);
    }
}
