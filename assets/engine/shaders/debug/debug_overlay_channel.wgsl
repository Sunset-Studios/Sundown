#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

struct ChannelConfig {
    channel_mask: vec4<u32>, // Which channels to display (1.0 = show, 0.0 = hide)
};

@group(1) @binding(0) var debug_texture: texture_2d<f32>;
@group(1) @binding(1) var<uniform> channel_config: ChannelConfig;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let val = textureSample(debug_texture, non_filtering_sampler, input.uv);
    let channels = channel_config.channel_mask;
    
    // Apply channel mask
    var masked_val = val * vec4<f32>(channels);

    // If only one channel in the mask is 1.0, show it as grayscale
    let num_set = channels.x + channels.y + channels.z + channels.w;
    if (num_set == 1u) {
        let value = masked_val.r + masked_val.g + masked_val.b + masked_val.a;
        masked_val = vec4<f32>(value, value, value, 1.0);
    }
    
    return vec4<f32>(masked_val.rgb, 1.0);
}

