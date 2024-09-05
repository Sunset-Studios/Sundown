import { vec3, vec4 } from 'gl-matrix';

export function screen_pos_to_world_pos(view_data, screen_x, screen_y, width, height, depth = 0) {
    // Convert screen coordinates to clip space
    const clip_x = (2.0 * screen_x / width) - 1.0;
    const clip_y = 1.0 - (2.0 * screen_y / height);
    
    // Create near and far points in clip space
    const clip_near = vec4.fromValues(clip_x, clip_y, -1.0, 1.0);
    const clip_far = vec4.fromValues(clip_x, clip_y, 1.0, 1.0);
    
    // Transform to world space
    const inv_vp_matrix = view_data.inverse_view_projection_matrix;
    const world_near = vec4.transformMat4(vec4.create(), clip_near, inv_vp_matrix);
    const world_far = vec4.transformMat4(vec4.create(), clip_far, inv_vp_matrix);
    
    // Perform perspective division
    vec4.scale(world_near, world_near, 1 / world_near[3]);
    vec4.scale(world_far, world_far, 1 / world_far[3]);
    
    // Calculate ray direction
    const ray_direction = vec3.subtract(vec3.create(), world_far, world_near);
    vec3.normalize(ray_direction, ray_direction);
    
    // Calculate intersection with plane at specified depth
    const camera_pos = view_data.position;
    const view_direction = vec3.fromValues(
        view_data.view_forward[0],
        view_data.view_forward[1],
        view_data.view_forward[2]
    );
    const t = depth / vec3.dot(ray_direction, view_direction);
    
    // Calculate final world position
    const world_pos = vec3.create();
    vec3.scaleAndAdd(world_pos, camera_pos, ray_direction, t);
    
    return world_pos;
}