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