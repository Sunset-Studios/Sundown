#include "common.wgsl"

// Particle render: writes each particle as a point into color and depth storage textures
// Assumes positions are in normalized device coordinates (x,y ∈ [-1,1], z ∈ [0,1])

@group(1) @binding(0) var<storage, read> positions: array<vec4f>;
@group(1) @binding(1) var<storage, read> indices: array<u32>;
@group(1) @binding(2) var out_color: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var in_depth: texture_2d<f32>;

@compute @workgroup_size(256)
fn cs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i: u32 = global_id.x;
    if (i >= arrayLength(&indices)) {
        return;
    }

    let pid = indices[i];
    let clip = positions[pid];   // clip.xyz in NDC, clip.w unused
    let ndc = clip.xyz;

    // Map from NDC [-1,1] to texture coords [0, width) / [0, height)
    let dims = textureDimensions(out_color);
    let uv = (ndc.xy * 0.5 + vec2f(0.5, 0.5)) * vec2f(f32(dims.x), f32(dims.y));

    // coord is vec2<u32> of pixel indices
    let d = textureLoad(in_depth, vec2<u32>(uv), 0).r;

    // Perform depth test: only write if current particle is closer
    if (ndc.z >= d) {
        return;
    }

    // Simple color: orange
    textureStore(out_color, vec2<u32>(uv), vec4f(1.0, 0.5, 0.0, 1.0));
} 