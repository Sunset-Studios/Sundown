#include "common.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) instance_index: u32,
};

@group(1) @binding(0) var motion_texture: texture_2d<f32>;
@group(1) @binding(1) var depth_texture: texture_2d<f32>;
@group(1) @binding(2) var position_texture: texture_2d<f32>;
@group(1) @binding(3) var scene_color: texture_2d<f32>;

const TILE_SIZE = 16.0;
const LINE_WIDTH = 1.0;
const MOTION_SCALE = 1.0;
const LINE_COLOR = vec3<f32>(1.0, 1.0, 0.0); // Yellow

// Distance from point to line segment
fn distance_to_line_segment(p: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
    let line_vec = end - start;
    let line_len_sq = dot(line_vec, line_vec);
    
    if (line_len_sq < 0.0001) {
        return distance(p, start);
    }
    
    let t = clamp(dot(p - start, line_vec) / line_len_sq, 0.0, 1.0);
    let projection = start + t * line_vec;
    return distance(p, projection);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let resolution = vec2<f32>(textureDimensions(motion_texture));
    let pixel_coord = input.uv * resolution;
    
    // Calculate which tile this pixel belongs to
    let tile_coord = floor(pixel_coord / TILE_SIZE);
    
    // Calculate the center of the tile
    let tile_center_pixel = (tile_coord + 0.5) * TILE_SIZE;
    let tile_center_uv = tile_center_pixel / resolution;
    
    // Sample NDC-space velocity (xy stored in motion texture)
    let motion_sample = textureSample(motion_texture, non_filtering_sampler, tile_center_uv);
    let ndc_velocity = motion_sample.xy;
    
    // Sample depth to check if this is valid geometry
    let depth = textureSample(depth_texture, non_filtering_sampler, tile_center_uv).r;
    
    // Convert NDC velocity directly to pixel-space motion
    let motion_pixels = ndc_velocity * resolution * vec2<f32>(0.5, -0.5) * MOTION_SCALE;
    
    // Calculate line endpoints in pixel space
    let line_start = tile_center_pixel;
    let line_end = tile_center_pixel + motion_pixels;
    
    // Check if motion is significant and geometry is valid
    let motion_magnitude = length(motion_pixels);
    let is_valid = motion_magnitude > 0.1 && depth < 0.9999;
    
    // Calculate distance from current pixel to the motion line
    let dist_to_line = distance_to_line_segment(pixel_coord, line_start, line_end);
    
    // Draw the line if we're close enough
    let on_line = dist_to_line < LINE_WIDTH && is_valid;
    
    // Draw an arrowhead at the end
    let to_end = pixel_coord - line_end;
    let dist_to_end = length(to_end);
    let arrow_size = 2.0;
    let on_arrowhead = dist_to_end < arrow_size && is_valid;
    
    // Sample scene color for background
    let background = textureSample(scene_color, non_filtering_sampler, input.uv);
    
    // Blend: show lines over dimmed scene
    let output = select(vec4<f32>(background.rgb * 0.7, 1.0), vec4<f32>(LINE_COLOR, 1.0), on_line || on_arrowhead);
    return output;
}

