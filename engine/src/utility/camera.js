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
    const t = depth / vec3.dot(ray_direction, view_data.view_forward);
    
    // Calculate final world position
    const world_pos = vec3.create();
    vec3.scaleAndAdd(world_pos, camera_pos, ray_direction, t);
    
    return world_pos;
}

/**
 * Calculates the world position corresponding to a screen coordinate by intersecting
 * the camera ray with one of the major axis planes (XY, XZ, or YZ).
 *
 * @param {object} view_data - Camera view data containing matrices and position.
 * @param {number} screen_x - The x-coordinate on the screen.
 * @param {number} screen_y - The y-coordinate on the screen.
 * @param {number} width - The width of the viewport/canvas.
 * @param {number} height - The height of the viewport/canvas.
 * @param {'x' | 'y' | 'z'} plane_axis - The axis normal to the desired plane ('x' for YZ, 'y' for XZ, 'z' for XY). Defaults to 'y'.
 * @returns {vec3 | null} The world position on the specified plane, or null if the ray is parallel to the plane.
 */
export function screen_pos_to_axis_plane_pos(
  view_data,
  screen_x,
  screen_y,
  width,
  height,
  plane_axis = 'y' // Default to 'y' for the XZ plane
) {
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
    vec4.scale(world_near, world_near, 1.0 / world_near[3]);
    vec4.scale(world_far, world_far, 1.0 / world_far[3]);

    // Ray origin is the camera position (effectively world_near after division)
    const ray_origin = vec3.fromValues(world_near[0], world_near[1], world_near[2]);
    // Calculate ray direction
    const ray_direction = vec3.subtract(vec3.create(), world_far, world_near);
    vec3.normalize(ray_direction, ray_direction);

    // Define the plane based on the axis
    let plane_normal;
    let axis_index;
    switch (plane_axis) {
        case 'x': // YZ plane (X=0)
            plane_normal = vec3.fromValues(1.0, 0.0, 0.0);
            axis_index = 0;
            break;
        case 'z': // XY plane (Z=0)
            plane_normal = vec3.fromValues(0.0, 0.0, 1.0);
            axis_index = 2;
            break;
        case 'y': // XZ plane (Y=0) - Default
        default:
            plane_normal = vec3.fromValues(0.0, 1.0, 0.0);
            axis_index = 1;
            break;
    }
    const point_on_plane = vec3.fromValues(0.0, 0.0, 0.0); // Origin lies on all major planes

    // Calculate the denominator for the intersection formula: dot(ray_direction, plane_normal)
    const denominator = vec3.dot(ray_direction, plane_normal);

    // Check if the ray is parallel to the plane (or very close to parallel)
    if (Math.abs(denominator) < 1e-6) {
        console.warn(`Ray is parallel to the plane normal to ${plane_axis}, cannot find intersection.`);
        // Return null to indicate failure
        return null;
    }

    // Calculate the numerator: dot(point_on_plane - ray_origin, plane_normal)
    const p0_l0 = vec3.subtract(vec3.create(), point_on_plane, ray_origin);
    const numerator = vec3.dot(p0_l0, plane_normal);

    // Calculate t, the distance along the ray to the intersection point
    const t = numerator / denominator;

    // Calculate the final world position on the plane
    const world_pos = vec3.create();
    vec3.scaleAndAdd(world_pos, ray_origin, ray_direction, t);

    // Ensure the coordinate corresponding to the plane's normal is exactly 0
    world_pos[axis_index] = 0.0;

    return world_pos;
}

export function world_pos_to_screen_pos(view_data, world_pos, width, height) {
    // Create a vec4 from the world position with w=1
    const world_point = vec4.fromValues(world_pos[0], world_pos[1], world_pos[2], 1.0);
    
    // Transform world position to clip space using the view-projection matrix
    const clip_pos = vec4.create();
    vec4.transformMat4(clip_pos, world_point, view_data.view_projection_matrix);
    
    // Early exit if the point is behind the camera
    if (clip_pos[3] <= 0) {
        return null;
    }
    
    // Perform perspective division to get normalized device coordinates
    const w_inv = 1.0 / clip_pos[3];
    const ndc_x = clip_pos[0] * w_inv;
    const ndc_y = clip_pos[1] * w_inv;
    const ndc_z = clip_pos[2] * w_inv;
    
    // Convert from NDC to screen coordinates
    const screen_x = ((ndc_x + 1.0) * 0.5) * width;
    const screen_y = ((1.0 - ndc_y) * 0.5) * height;
    
    // Return screen coordinates and visibility information
    return {
        x: screen_x,
        y: screen_y,
        is_visible: (ndc_z >= -1.0 && ndc_z <= 1.0)
    };
}