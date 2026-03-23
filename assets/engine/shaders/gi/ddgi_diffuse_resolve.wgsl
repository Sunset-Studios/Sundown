#include "common.wgsl"

const DDGI_RESOLVE_SKY_NORMAL_LENGTH_SQ_EPS: f32 = 1e-10;
const DDGI_RESOLVE_DEPTH_SIGMA: f32 = 0.01;
const DDGI_RESOLVE_NORMAL_POWER: f32 = 32.0;

@group(1) @binding(0) var input_diffuse_low: texture_2d<f32>;
@group(1) @binding(1) var hzb_texture: texture_2d<f32>;
@group(1) @binding(2) var gbuffer_normal: texture_2d<f32>;
@group(1) @binding(3) var output_diffuse: texture_storage_2d<rgba16float, write>;

fn clamp_i32(v: vec2<i32>, lo: vec2<i32>, hi: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y));
}

fn ddgi_low_res_to_full_res_coord(
    low_coord: vec2<u32>,
    low_res: vec2<u32>,
    full_res: vec2<u32>
) -> vec2<i32> {
    let uv = (vec2<f32>(low_coord) + vec2<f32>(0.5)) / vec2<f32>(f32(low_res.x), f32(low_res.y));
    let full_coord = min(
        vec2<u32>(uv * vec2<f32>(f32(full_res.x), f32(full_res.y))),
        full_res - vec2<u32>(1u)
    );
    return vec2<i32>(i32(full_coord.x), i32(full_coord.y));
}

fn ddgi_resolve_edge_weight(
    center_depth: f32,
    center_normal: vec3<f32>,
    sample_depth: f32,
    sample_normal: vec3<f32>
) -> f32 {
    let depth_diff = abs(sample_depth - center_depth) / max(abs(center_depth), 1e-4);
    let depth_weight =
        exp(-(depth_diff * depth_diff) / (2.0 * DDGI_RESOLVE_DEPTH_SIGMA * DDGI_RESOLVE_DEPTH_SIGMA));
    let normal_weight = pow(max(dot(center_normal, sample_normal), 0.0), DDGI_RESOLVE_NORMAL_POWER);
    return depth_weight * normal_weight;
}

@compute @workgroup_size(8, 8, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_res = textureDimensions(output_diffuse);
    if (gid.x >= full_res.x || gid.y >= full_res.y) {
        return;
    }

    let low_res = textureDimensions(input_diffuse_low);
    let full_pixel_coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let center_normal_data = textureLoad(gbuffer_normal, full_pixel_coord, 0);

    if (dot(center_normal_data.xyz, center_normal_data.xyz) <= DDGI_RESOLVE_SKY_NORMAL_LENGTH_SQ_EPS) {
        textureStore(output_diffuse, full_pixel_coord, vec4<f32>(0.0));
        return;
    }

    let center_normal = safe_normalize(center_normal_data.xyz);
    let center_depth = textureLoad(hzb_texture, full_pixel_coord, 0).r;
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(full_res.x), f32(full_res.y));
    let low_pos = uv * vec2<f32>(f32(low_res.x), f32(low_res.y)) - vec2<f32>(0.5);
    let base_i = vec2<i32>(floor(low_pos));
    let frac = low_pos - vec2<f32>(base_i);

    let max_low_i = vec2<i32>(i32(low_res.x) - 1, i32(low_res.y) - 1);
    let c00_i = clamp_i32(base_i, vec2<i32>(0), max_low_i);
    let c10_i = clamp_i32(base_i + vec2<i32>(1, 0), vec2<i32>(0), max_low_i);
    let c01_i = clamp_i32(base_i + vec2<i32>(0, 1), vec2<i32>(0), max_low_i);
    let c11_i = clamp_i32(base_i + vec2<i32>(1, 1), vec2<i32>(0), max_low_i);

    let w00 = (1.0 - frac.x) * (1.0 - frac.y);
    let w10 = frac.x * (1.0 - frac.y);
    let w01 = (1.0 - frac.x) * frac.y;
    let w11 = frac.x * frac.y;

    let c00_full = ddgi_low_res_to_full_res_coord(vec2<u32>(c00_i), low_res, full_res);
    let c10_full = ddgi_low_res_to_full_res_coord(vec2<u32>(c10_i), low_res, full_res);
    let c01_full = ddgi_low_res_to_full_res_coord(vec2<u32>(c01_i), low_res, full_res);
    let c11_full = ddgi_low_res_to_full_res_coord(vec2<u32>(c11_i), low_res, full_res);

    let c00_normal_data = textureLoad(gbuffer_normal, c00_full, 0);
    let c10_normal_data = textureLoad(gbuffer_normal, c10_full, 0);
    let c01_normal_data = textureLoad(gbuffer_normal, c01_full, 0);
    let c11_normal_data = textureLoad(gbuffer_normal, c11_full, 0);

    let e00 = select(
        0.0,
        ddgi_resolve_edge_weight(
            center_depth,
            center_normal,
            textureLoad(hzb_texture, c00_full, 0).r,
            safe_normalize(c00_normal_data.xyz)
        ),
        dot(c00_normal_data.xyz, c00_normal_data.xyz) > DDGI_RESOLVE_SKY_NORMAL_LENGTH_SQ_EPS
    );
    let e10 = select(
        0.0,
        ddgi_resolve_edge_weight(
            center_depth,
            center_normal,
            textureLoad(hzb_texture, c10_full, 0).r,
            safe_normalize(c10_normal_data.xyz)
        ),
        dot(c10_normal_data.xyz, c10_normal_data.xyz) > DDGI_RESOLVE_SKY_NORMAL_LENGTH_SQ_EPS
    );
    let e01 = select(
        0.0,
        ddgi_resolve_edge_weight(
            center_depth,
            center_normal,
            textureLoad(hzb_texture, c01_full, 0).r,
            safe_normalize(c01_normal_data.xyz)
        ),
        dot(c01_normal_data.xyz, c01_normal_data.xyz) > DDGI_RESOLVE_SKY_NORMAL_LENGTH_SQ_EPS
    );
    let e11 = select(
        0.0,
        ddgi_resolve_edge_weight(
            center_depth,
            center_normal,
            textureLoad(hzb_texture, c11_full, 0).r,
            safe_normalize(c11_normal_data.xyz)
        ),
        dot(c11_normal_data.xyz, c11_normal_data.xyz) > DDGI_RESOLVE_SKY_NORMAL_LENGTH_SQ_EPS
    );

    let ww00 = w00 * e00;
    let ww10 = w10 * e10;
    let ww01 = w01 * e01;
    let ww11 = w11 * e11;
    let weight_sum = ww00 + ww10 + ww01 + ww11;
    let inv_weight_sum = 1.0 / max(weight_sum, 1e-6);
    let use_edge_weights = weight_sum > 1e-6;

    let fw00 = select(w00, ww00 * inv_weight_sum, use_edge_weights);
    let fw10 = select(w10, ww10 * inv_weight_sum, use_edge_weights);
    let fw01 = select(w01, ww01 * inv_weight_sum, use_edge_weights);
    let fw11 = select(w11, ww11 * inv_weight_sum, use_edge_weights);

    let s00 = textureLoad(input_diffuse_low, c00_i, 0);
    let s10 = textureLoad(input_diffuse_low, c10_i, 0);
    let s01 = textureLoad(input_diffuse_low, c01_i, 0);
    let s11 = textureLoad(input_diffuse_low, c11_i, 0);

    textureStore(
        output_diffuse,
        full_pixel_coord,
        s00 * fw00 + s10 * fw10 + s01 * fw01 + s11 * fw11
    );
}
