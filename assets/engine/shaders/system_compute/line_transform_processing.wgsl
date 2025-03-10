#include "common.wgsl"

// Line transform processing compute shader
// Calculates transform matrices for lines based on start and end positions

// Define the line position struct
struct LinePosition {
    start: vec4f,
    end: vec4f,
}

// Define the binding groups
@group(1) @binding(0) var<storage, read_write> transforms: array<mat4x4f>;
@group(1) @binding(1) var<storage, read> line_positions: array<LinePosition>;

fn mat4_identity() -> mat4x4f {
    return mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0)
    );
}

fn mat4_from_scaling(scale: vec3f) -> mat4x4f {
    return mat4x4f(
        vec4f(scale.x, 0.0, 0.0, 0.0),
        vec4f(0.0, scale.y, 0.0, 0.0),
        vec4f(0.0, 0.0, scale.z, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0)
    );
}

// Main function to create a line transform
fn create_line_transform(start: vec4f, end: vec4f) -> mat4x4f {
    // Calculate direction vector from start to end
    let direction = end.xyz - start.xyz;
    
    // Calculate length of the line
    let length = sqrt(dot(direction, direction));
    
    if (length < 0.0001) {
        // Handle degenerate case (zero-length line)
        return mat4_identity();
    }
    
    // Normalized direction
    let normalized_direction = direction / length;
    
    // Find perpendicular vectors to create a coordinate system
    // Start with a default up vector
    var up = vec3f(0.0, 1.0, 0.0);
    
    // If direction is too close to up, use a different reference vector
    if (abs(dot(normalized_direction, up)) > 0.99) {
        up = vec3f(0.0, 0.0, 1.0);
    }
    
    // Calculate right vector (perpendicular to both direction and up)
    let right = normalize(cross(normalized_direction, up));
    
    // Recalculate a true up vector that's perpendicular to both direction and right
    let true_up = normalize(cross(right, normalized_direction));
    
    // Build rotation matrix (column-major)
    // The columns represent the transformed basis vectors
    var rotation_matrix = mat4x4f(
        vec4f(normalized_direction, 0.0),  // x-axis maps to direction
        vec4f(true_up, 0.0),               // y-axis maps to up
        vec4f(right, 0.0),                 // z-axis maps to right
        vec4f(0.0, 0.0, 0.0, 1.0)          // no translation in rotation matrix
    );
    
    // Create translation matrix to position at start point
    let translation_matrix = mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(start.xyz, 1.0)
    );
    
    // Apply scale to make the line the correct length
    let scale_matrix = mat4_from_scaling(vec3f(length, 1.0, 1.0));
    
    // Combine transformations: first scale, then rotate, then translate
    // Order matters in matrix multiplication
    return translation_matrix * rotation_matrix * scale_matrix;
}

// Compute shader entry point
@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x;
    
    // Make sure we don't go out of bounds
    if (index >= arrayLength(&line_positions)) {
        return;
    }
    
    // Get the line position
    let line_pos = line_positions[index];

    let line_transform = create_line_transform(line_pos.start, line_pos.end);
    
    // Calculate the transform
    transforms[index] = line_transform;
}