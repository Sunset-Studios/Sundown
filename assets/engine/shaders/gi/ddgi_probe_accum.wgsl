#include "common.wgsl"

@group(1) @binding(0) var probe_cubemap   : texture_2d_array<f32>;
@group(1) @binding(1) var<uniform> gi_params     : GIParams;
@group(1) @binding(2) var<storage, write> gi_irradiance : texture_storage_3d<rgba16float, write>;
@group(1) @binding(3) var<storage, write> gi_depth      : texture_storage_3d<r32float,   write>;
@group(1) @binding(4) var linear_sampler   : sampler;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // single thread handles one probe
    let probe_idx = gi_params.current_index;
    let dims     = gi_params.dims;
    let rem      = probe_idx % (dims.x * dims.y);
    let x        = rem % dims.x;
    let y        = rem / dims.x;
    let z        = probe_idx / (dims.x * dims.y);

    // average the six cubemap faces at texel center
    var accum = vec3<f32>(0.0);
    for (var face: u32 = 0u; face < 6u; face = face + 1u) {
        let uv_layer = vec3<f32>(0.5, 0.5, f32(face));
        accum += textureSample(probe_cubemap, linear_sampler, uv_layer).rgb;
    }
    let irradiance = accum * (1.0 / 6.0);

    // write into the 3D irradiance volume
    textureStore(
      gi_irradiance,
      vec3<i32>(i32(x), i32(y), i32(z)),
      vec4<f32>(irradiance, 1.0)
    );

    // placeholder depth = probe radius
    let radius = max(max(gi_params.spacing.x, gi_params.spacing.y), gi_params.spacing.z);
    textureStore(
      gi_depth,
      vec3<i32>(i32(x), i32(y), i32(z)),
      vec4<f32>(radius, 0.0, 0.0, 0.0)
    );
} 