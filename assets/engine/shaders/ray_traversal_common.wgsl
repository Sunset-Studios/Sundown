// Shared ray traversal interface.
//
// Strategy headers implement these entry points:
// - trace_ray_closest(ray) returns the nearest surface hit or a miss.
// - trace_ray_any(ray) returns true when any surface blocks the ray.
// - trace_ray_closest_tlas(ray, tlas_only) is for low-level callers that can
//   intentionally stop at the TLAS instance bounds.

