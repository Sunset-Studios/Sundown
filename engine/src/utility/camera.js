import { vec4 } from 'gl-matrix';

export function screen_pos_to_world_pos(view_data, screen_x, screen_y, depth = 0) {
    const { inverse_view_projection_matrix } = view_data;

    // Convert screen coordinates to normalized device coordinates (NDC)
    const ndc_x = (screen_x / window.innerWidth) * 2 - 1;
    const ndc_y = 1 - (screen_y / window.innerHeight) * 2; // Flip Y-axis
    const ndc_z = depth * 2 - 1; // Convert depth to NDC space

    // Create a vector in NDC space
    const ndc_vector = vec4.fromValues(ndc_x, ndc_y, ndc_z, 1);

    // Transform NDC to world space
    const world_vector = vec4.create();
    vec4.transformMat4(world_vector, ndc_vector, inverse_view_projection_matrix);

    // Perform perspective divide
    const w = world_vector[3];
    const world_position = vec4.fromValues(
        world_vector[0] / w,
        world_vector[1] / w,
        world_vector[2] / w,
        world_vector[3] / w
    );

    return world_position;
}