/**
 * Returns an euler angle representation of a quaternion, in degrees
 * @param  {vec3} out Euler angles, pitch-yaw-roll
 * @param  {quat} mat Quaternion
 * @return {vec3} out
 */
export function quat_to_euler(out, q) {
  // Assuming q is in the form [x, y, z, w]
  const [x, y, z, w] = q;

  // Calculate pitch (x-axis rotation)
  const sinp = 2.0 * (w * y - z * x);
  if (Math.abs(sinp) >= 1) {
    out[1] = Math.copySign(Math.PI / 2, sinp); // use 90 degrees if out of range
  } else {
    out[0] = Math.asin(sinp);
  }

  // Calculate yaw (y-axis rotation)
  const siny_cosp = 2.0 * (w * z + x * y);
  const cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  out[1] = Math.atan2(siny_cosp, cosy_cosp);

  // Calculate roll (z-axis rotation)
  const sinr_cosp = 2.0 * (w * x + y * z);
  const cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  out[2] = Math.atan2(sinr_cosp, cosr_cosp);

  // Convert to degrees
  out[0] *= (180 / Math.PI);
  out[1] *= (180 / Math.PI);
  out[2] *= (180 / Math.PI);

  return out; // [pitch, yaw, roll] in degrees
}

export function direction_vector_to_quat(vec) {
    // This assumes vec is normalized
    const up = [0, 1, 0];
    const right = [0, 0, 0];
    const forward = [-vec[0], -vec[1], -vec[2]];
    
    // Calculate right vector using cross product of up and forward
    right[0] = up[1] * forward[2] - up[2] * forward[1];
    right[1] = up[2] * forward[0] - up[0] * forward[2];
    right[2] = up[0] * forward[1] - up[1] * forward[0];
    
    // Normalize right vector
    const right_length = Math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right[0] /= right_length;
    right[1] /= right_length;
    right[2] /= right_length;
    
    // Recalculate up vector to ensure orthogonality
    up[0] = forward[1] * right[2] - forward[2] * right[1];
    up[1] = forward[2] * right[0] - forward[0] * right[2];
    up[2] = forward[0] * right[1] - forward[1] * right[0];
    
    // Convert to quaternion (using rotation matrix to quaternion conversion)
    const trace = right[0] + up[1] + forward[2];
    let qw, qx, qy, qz;
    
    if (trace > 0) {
        const S = Math.sqrt(trace + 1.0) * 2;
        qw = 0.25 * S;
        qx = (up[2] - forward[1]) / S;
        qy = (forward[0] - right[2]) / S;
        qz = (right[1] - up[0]) / S;
    } else if (right[0] > up[1] && right[0] > forward[2]) {
        const S = Math.sqrt(1.0 + right[0] - up[1] - forward[2]) * 2;
        qw = (up[2] - forward[1]) / S;
        qx = 0.25 * S;
        qy = (right[1] + up[0]) / S;
        qz = (right[2] + forward[0]) / S;
    } else if (up[1] > forward[2]) {
        const S = Math.sqrt(1.0 + up[1] - right[0] - forward[2]) * 2;
        qw = (forward[0] - right[2]) / S;
        qx = (right[1] + up[0]) / S;
        qy = 0.25 * S;
        qz = (up[2] + forward[1]) / S;
    } else {
        const S = Math.sqrt(1.0 + forward[2] - right[0] - up[1]) * 2;
        qw = (right[1] - up[0]) / S;
        qx = (right[2] + forward[0]) / S;
        qy = (up[2] + forward[1]) / S;
        qz = 0.25 * S;
    }
    
    return [qx, qy, qz, qw];
}
