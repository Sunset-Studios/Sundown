// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║               SPHERICAL HARMONICS SUPPORT LIBRARY FOR WGSL                ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  Based on SHforHLSL by MJP (https://github.com/TheRealMJP/SHforHLSL)      ║
// ║  Adapted for WGSL.                                                        ║
// ║                                                                           ║
// ║  This library implements types and utility functions for working with     ║
// ║  low-order spherical harmonics, focused on use cases for graphics:        ║
// ║                                                                           ║
// ║  • L1 (2 bands, 4 coefficients)                                           ║
// ║  • L2 (3 bands, 9 coefficients)                                           ║
// ║  • Scalar and RGB variants                                                ║
// ║  • Projection, evaluation, convolution, rotation                          ║
// ║  • Irradiance calculation for diffuse lighting                            ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// =============================================================================
// CONSTANTS
// =============================================================================

const SH_PI: f32 = 3.141592654;
const SH_SQRT_PI: f32 = 1.7724538509;  // sqrt(PI)

// ─────────────────────────────────────────────────────────────────────────────
// Zonal Harmonic coefficients for cosine lobe convolution
// Used when convolving radiance to compute irradiance
// ─────────────────────────────────────────────────────────────────────────────
const SH_COSINE_A0: f32 = 3.141592654;                // PI
const SH_COSINE_A1: f32 = 2.094395102;                // (2 * PI) / 3
const SH_COSINE_A2: f32 = 0.785398163;                // PI / 4

// ─────────────────────────────────────────────────────────────────────────────
// SH Basis function normalization constants
// These ensure orthonormality of the spherical harmonic basis functions
// ─────────────────────────────────────────────────────────────────────────────
const SH_BASIS_L0: f32 = 0.282094792;                 // 1 / (2 * sqrt(PI))
const SH_BASIS_L1: f32 = 0.488602512;                 // sqrt(3) / (2 * sqrt(PI))
const SH_BASIS_L2_MN2: f32 = 1.092548431;             // sqrt(15) / (2 * sqrt(PI))
const SH_BASIS_L2_MN1: f32 = 1.092548431;             // sqrt(15) / (2 * sqrt(PI))
const SH_BASIS_L2_M0: f32 = 0.315391565;              // sqrt(5) / (4 * sqrt(PI))
const SH_BASIS_L2_M1: f32 = 1.092548431;              // sqrt(15) / (2 * sqrt(PI))
const SH_BASIS_L2_M2: f32 = 0.546274215;              // sqrt(15) / (4 * sqrt(PI))

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                           SH DATA STRUCTURES                              ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// L1 Spherical Harmonics (2 bands, 4 coefficients)
// Scalar version - single channel
// ─────────────────────────────────────────────────────────────────────────────
struct SH_L1 {
    c: array<f32, 4>,
}

// ─────────────────────────────────────────────────────────────────────────────
// L1 Spherical Harmonics (2 bands, 4 coefficients)  
// RGB version - three channels for color
// ─────────────────────────────────────────────────────────────────────────────
struct SH_L1_RGB {
    c: array<vec3<f32>, 4>,
}

// ─────────────────────────────────────────────────────────────────────────────
// L2 Spherical Harmonics (3 bands, 9 coefficients)
// Scalar version - single channel
// ─────────────────────────────────────────────────────────────────────────────
struct SH_L2 {
    c: array<f32, 9>,
}

// ─────────────────────────────────────────────────────────────────────────────
// L2 Spherical Harmonics (3 bands, 9 coefficients)
// RGB version - three channels for color
// ─────────────────────────────────────────────────────────────────────────────
struct SH_L2_RGB {
    c: array<vec3<f32>, 9>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Packed L1 (for storage optimization)
// Uses f16 packing: 4 coefficients → 2 u32 values
// ─────────────────────────────────────────────────────────────────────────────
struct SH_L1_Packed {
    data: array<u32, 2>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Packed L1 RGB (for storage optimization)
// Uses f16 packing: 12 floats → 6 u32 values
// ─────────────────────────────────────────────────────────────────────────────
struct SH_L1_RGB_Packed {
    data: array<u32, 6>,
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                         INITIALIZATION FUNCTIONS                          ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Create a zeroed L1 SH
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_zero() -> SH_L1 {
    var sh: SH_L1;
    sh.c[0] = 0.0;
    sh.c[1] = 0.0;
    sh.c[2] = 0.0;
    sh.c[3] = 0.0;
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create an L1 SH with explicit coefficients
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_create(c0: f32, c1: f32, c2: f32, c3: f32) -> SH_L1 {
    var sh: SH_L1;
    sh.c[0] = c0;
    sh.c[1] = c1;
    sh.c[2] = c2;
    sh.c[3] = c3;
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create a zeroed L1 RGB SH
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_rgb_zero() -> SH_L1_RGB {
    var sh: SH_L1_RGB;
    sh.c[0] = vec3<f32>(0.0);
    sh.c[1] = vec3<f32>(0.0);
    sh.c[2] = vec3<f32>(0.0);
    sh.c[3] = vec3<f32>(0.0);
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create a zeroed L2 SH
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l2_zero() -> SH_L2 {
    var sh: SH_L2;
    for (var i = 0u; i < 9u; i = i + 1u) {
        sh.c[i] = 0.0;
    }
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Create a zeroed L2 RGB SH
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l2_rgb_zero() -> SH_L2_RGB {
    var sh: SH_L2_RGB;
    for (var i = 0u; i < 9u; i = i + 1u) {
        sh.c[i] = vec3<f32>(0.0);
    }
    return sh;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                         PACKING / UNPACKING                               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Pack L1 SH coefficients into half-precision storage
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_pack(sh: SH_L1) -> SH_L1_Packed {
    var packed: SH_L1_Packed;
    packed.data[0] = pack2x16float(vec2<f32>(sh.c[0], sh.c[1]));
    packed.data[1] = pack2x16float(vec2<f32>(sh.c[2], sh.c[3]));
    return packed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unpack L1 SH coefficients from half-precision storage
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_unpack(packed: SH_L1_Packed) -> SH_L1 {
    var sh: SH_L1;
    let pair0 = unpack2x16float(packed.data[0]);
    let pair1 = unpack2x16float(packed.data[1]);
    sh.c[0] = pair0.x;
    sh.c[1] = pair0.y;
    sh.c[2] = pair1.x;
    sh.c[3] = pair1.y;
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pack L1 RGB SH coefficients into half-precision storage
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_rgb_pack(sh: SH_L1_RGB) -> SH_L1_RGB_Packed {
    var packed: SH_L1_RGB_Packed;
    packed.data[0] = pack2x16float(vec2<f32>(sh.c[0].x, sh.c[0].y));
    packed.data[1] = pack2x16float(vec2<f32>(sh.c[0].z, sh.c[1].x));
    packed.data[2] = pack2x16float(vec2<f32>(sh.c[1].y, sh.c[1].z));
    packed.data[3] = pack2x16float(vec2<f32>(sh.c[2].x, sh.c[2].y));
    packed.data[4] = pack2x16float(vec2<f32>(sh.c[2].z, sh.c[3].x));
    packed.data[5] = pack2x16float(vec2<f32>(sh.c[3].y, sh.c[3].z));
    return packed;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unpack L1 RGB SH coefficients from half-precision storage
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_rgb_unpack(packed: SH_L1_RGB_Packed) -> SH_L1_RGB {
    var sh: SH_L1_RGB;
    let p0 = unpack2x16float(packed.data[0]);
    let p1 = unpack2x16float(packed.data[1]);
    let p2 = unpack2x16float(packed.data[2]);
    let p3 = unpack2x16float(packed.data[3]);
    let p4 = unpack2x16float(packed.data[4]);
    let p5 = unpack2x16float(packed.data[5]);
    
    sh.c[0] = vec3<f32>(p0.x, p0.y, p1.x);
    sh.c[1] = vec3<f32>(p1.y, p2.x, p2.y);
    sh.c[2] = vec3<f32>(p3.x, p3.y, p4.x);
    sh.c[3] = vec3<f32>(p4.y, p5.x, p5.y);
    return sh;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                         ARITHMETIC OPERATIONS                             ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// ADD: Sum two sets of SH coefficients
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_add(a: SH_L1, b: SH_L1) -> SH_L1 {
    var result: SH_L1;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] + b.c[i];
    }
    return result;
}

fn sh_l1_rgb_add(a: SH_L1_RGB, b: SH_L1_RGB) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] + b.c[i];
    }
    return result;
}

fn sh_l2_add(a: SH_L2, b: SH_L2) -> SH_L2 {
    var result: SH_L2;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] + b.c[i];
    }
    return result;
}

fn sh_l2_rgb_add(a: SH_L2_RGB, b: SH_L2_RGB) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] + b.c[i];
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// SUBTRACT: Difference of two sets of SH coefficients
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_subtract(a: SH_L1, b: SH_L1) -> SH_L1 {
    var result: SH_L1;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] - b.c[i];
    }
    return result;
}

fn sh_l1_rgb_subtract(a: SH_L1_RGB, b: SH_L1_RGB) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] - b.c[i];
    }
    return result;
}

fn sh_l2_subtract(a: SH_L2, b: SH_L2) -> SH_L2 {
    var result: SH_L2;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] - b.c[i];
    }
    return result;
}

fn sh_l2_rgb_subtract(a: SH_L2_RGB, b: SH_L2_RGB) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] - b.c[i];
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// MULTIPLY: Scale SH coefficients by a value
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_multiply(a: SH_L1, b: f32) -> SH_L1 {
    var result: SH_L1;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] * b;
    }
    return result;
}

fn sh_l1_rgb_multiply(a: SH_L1_RGB, b: vec3<f32>) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] * b;
    }
    return result;
}

fn sh_l1_rgb_multiply_scalar(a: SH_L1_RGB, b: f32) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] * b;
    }
    return result;
}

fn sh_l2_multiply(a: SH_L2, b: f32) -> SH_L2 {
    var result: SH_L2;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] * b;
    }
    return result;
}

fn sh_l2_rgb_multiply(a: SH_L2_RGB, b: vec3<f32>) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] * b;
    }
    return result;
}

fn sh_l2_rgb_multiply_scalar(a: SH_L2_RGB, b: f32) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] * b;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// DIVIDE: Scale SH coefficients by inverse of a value
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_divide(a: SH_L1, b: f32) -> SH_L1 {
    var result: SH_L1;
    let inv_b = 1.0 / b;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] * inv_b;
    }
    return result;
}

fn sh_l1_rgb_divide(a: SH_L1_RGB, b: vec3<f32>) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    let inv_b = 1.0 / b;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = a.c[i] * inv_b;
    }
    return result;
}

fn sh_l2_divide(a: SH_L2, b: f32) -> SH_L2 {
    var result: SH_L2;
    let inv_b = 1.0 / b;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] * inv_b;
    }
    return result;
}

fn sh_l2_rgb_divide(a: SH_L2_RGB, b: vec3<f32>) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    let inv_b = 1.0 / b;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = a.c[i] * inv_b;
    }
    return result;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                           CONVERSION FUNCTIONS                            ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Truncate L2 to L1 (discard higher-order coefficients)
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l2_to_l1(sh: SH_L2) -> SH_L1 {
    var result: SH_L1;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = sh.c[i];
    }
    return result;
}

fn sh_l2_rgb_to_l1(sh: SH_L2_RGB) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = sh.c[i];
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert scalar SH to RGB (broadcast to all channels)
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_to_rgb(sh: SH_L1) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result.c[i] = vec3<f32>(sh.c[i]);
    }
    return result;
}

fn sh_l2_to_rgb(sh: SH_L2) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result.c[i] = vec3<f32>(sh.c[i]);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear interpolation between SH coefficients
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_lerp(x: SH_L1, y: SH_L1, s: f32) -> SH_L1 {
    return sh_l1_add(sh_l1_multiply(x, 1.0 - s), sh_l1_multiply(y, s));
}

fn sh_l1_rgb_lerp(x: SH_L1_RGB, y: SH_L1_RGB, s: f32) -> SH_L1_RGB {
    return sh_l1_rgb_add(
        sh_l1_rgb_multiply_scalar(x, 1.0 - s),
        sh_l1_rgb_multiply_scalar(y, s)
    );
}

fn sh_l2_lerp(x: SH_L2, y: SH_L2, s: f32) -> SH_L2 {
    return sh_l2_add(sh_l2_multiply(x, 1.0 - s), sh_l2_multiply(y, s));
}

fn sh_l2_rgb_lerp(x: SH_L2_RGB, y: SH_L2_RGB, s: f32) -> SH_L2_RGB {
    return sh_l2_rgb_add(
        sh_l2_rgb_multiply_scalar(x, 1.0 - s),
        sh_l2_rgb_multiply_scalar(y, s)
    );
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                          PROJECTION FUNCTIONS                             ║
// ║                                                                           ║
// ║  Project a value in a direction onto SH basis functions.                  ║
// ║  Used when integrating radiance samples onto SH representation.           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Project onto L1 (scalar)
// ─────────────────────────────────────────────────────────────────────────────
fn sh_project_onto_l1(direction: vec3<f32>, value: f32) -> SH_L1 {
    var sh: SH_L1;
    
    // L0 band (constant/ambient term)
    sh.c[0] = SH_BASIS_L0 * value;
    
    // L1 band (linear/directional terms)
    sh.c[1] = SH_BASIS_L1 * direction.y * value;
    sh.c[2] = SH_BASIS_L1 * direction.z * value;
    sh.c[3] = SH_BASIS_L1 * direction.x * value;
    
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Project onto L1 (RGB)
// ─────────────────────────────────────────────────────────────────────────────
fn sh_project_onto_l1_rgb(direction: vec3<f32>, value: vec3<f32>) -> SH_L1_RGB {
    var sh: SH_L1_RGB;
    
    // L0 band (constant/ambient term)
    sh.c[0] = SH_BASIS_L0 * value;
    
    // L1 band (linear/directional terms)
    sh.c[1] = SH_BASIS_L1 * direction.y * value;
    sh.c[2] = SH_BASIS_L1 * direction.z * value;
    sh.c[3] = SH_BASIS_L1 * direction.x * value;
    
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Project onto L2 (scalar)
// ─────────────────────────────────────────────────────────────────────────────
fn sh_project_onto_l2(direction: vec3<f32>, value: f32) -> SH_L2 {
    var sh: SH_L2;
    
    // L0 band
    sh.c[0] = SH_BASIS_L0 * value;
    
    // L1 band
    sh.c[1] = SH_BASIS_L1 * direction.y * value;
    sh.c[2] = SH_BASIS_L1 * direction.z * value;
    sh.c[3] = SH_BASIS_L1 * direction.x * value;
    
    // L2 band (quadratic terms)
    sh.c[4] = SH_BASIS_L2_MN2 * direction.x * direction.y * value;
    sh.c[5] = SH_BASIS_L2_MN1 * direction.y * direction.z * value;
    sh.c[6] = SH_BASIS_L2_M0 * (3.0 * direction.z * direction.z - 1.0) * value;
    sh.c[7] = SH_BASIS_L2_M1 * direction.x * direction.z * value;
    sh.c[8] = SH_BASIS_L2_M2 * (direction.x * direction.x - direction.y * direction.y) * value;
    
    return sh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Project onto L2 (RGB)
// ─────────────────────────────────────────────────────────────────────────────
fn sh_project_onto_l2_rgb(direction: vec3<f32>, value: vec3<f32>) -> SH_L2_RGB {
    var sh: SH_L2_RGB;
    
    // L0 band
    sh.c[0] = SH_BASIS_L0 * value;
    
    // L1 band
    sh.c[1] = SH_BASIS_L1 * direction.y * value;
    sh.c[2] = SH_BASIS_L1 * direction.z * value;
    sh.c[3] = SH_BASIS_L1 * direction.x * value;
    
    // L2 band
    sh.c[4] = SH_BASIS_L2_MN2 * direction.x * direction.y * value;
    sh.c[5] = SH_BASIS_L2_MN1 * direction.y * direction.z * value;
    sh.c[6] = SH_BASIS_L2_M0 * (3.0 * direction.z * direction.z - 1.0) * value;
    sh.c[7] = SH_BASIS_L2_M1 * direction.x * direction.z * value;
    sh.c[8] = SH_BASIS_L2_M2 * (direction.x * direction.x - direction.y * direction.y) * value;
    
    return sh;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                          DOT PRODUCT FUNCTIONS                            ║
// ║                                                                           ║
// ║  Compute the inner product between two sets of SH coefficients.           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

fn sh_l1_dot(a: SH_L1, b: SH_L1) -> f32 {
    var result: f32 = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        result = result + a.c[i] * b.c[i];
    }
    return result;
}

fn sh_l1_rgb_dot(a: SH_L1_RGB, b: SH_L1_RGB) -> vec3<f32> {
    var result: vec3<f32> = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        result = result + a.c[i] * b.c[i];
    }
    return result;
}

fn sh_l2_dot(a: SH_L2, b: SH_L2) -> f32 {
    var result: f32 = 0.0;
    for (var i = 0u; i < 9u; i = i + 1u) {
        result = result + a.c[i] * b.c[i];
    }
    return result;
}

fn sh_l2_rgb_dot(a: SH_L2_RGB, b: SH_L2_RGB) -> vec3<f32> {
    var result: vec3<f32> = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        result = result + a.c[i] * b.c[i];
    }
    return result;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                          EVALUATION FUNCTIONS                             ║
// ║                                                                           ║
// ║  "Look up" a value from SH coefficients in a particular direction.        ║
// ║  Projects a delta in the direction and computes dot product with SH.      ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

fn sh_l1_evaluate(sh: SH_L1, direction: vec3<f32>) -> f32 {
    let projected_delta = sh_project_onto_l1(direction, 1.0);
    return sh_l1_dot(projected_delta, sh);
}

fn sh_l1_rgb_evaluate(sh: SH_L1_RGB, direction: vec3<f32>) -> vec3<f32> {
    let projected_delta = sh_project_onto_l1_rgb(direction, vec3<f32>(1.0));
    return sh_l1_rgb_dot(projected_delta, sh);
}

fn sh_l2_evaluate(sh: SH_L2, direction: vec3<f32>) -> f32 {
    let projected_delta = sh_project_onto_l2(direction, 1.0);
    return sh_l2_dot(projected_delta, sh);
}

fn sh_l2_rgb_evaluate(sh: SH_L2_RGB, direction: vec3<f32>) -> vec3<f32> {
    let projected_delta = sh_project_onto_l2_rgb(direction, vec3<f32>(1.0));
    return sh_l2_rgb_dot(projected_delta, sh);
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                       ZONAL HARMONIC CONVOLUTION                          ║
// ║                                                                           ║
// ║  Convolve SH with zonal harmonics (rotationally symmetric kernels).       ║
// ║  This is the key operation for computing irradiance from radiance.        ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Convolve L1 with zonal harmonics
// zh.x = L0 scale, zh.y = L1 scale
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_convolve_zh(sh: SH_L1, zh: vec2<f32>) -> SH_L1 {
    var result: SH_L1;
    
    // L0
    result.c[0] = sh.c[0] * zh.x;
    
    // L1
    result.c[1] = sh.c[1] * zh.y;
    result.c[2] = sh.c[2] * zh.y;
    result.c[3] = sh.c[3] * zh.y;
    
    return result;
}

fn sh_l1_rgb_convolve_zh(sh: SH_L1_RGB, zh: vec2<f32>) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    
    // L0
    result.c[0] = sh.c[0] * zh.x;
    
    // L1
    result.c[1] = sh.c[1] * zh.y;
    result.c[2] = sh.c[2] * zh.y;
    result.c[3] = sh.c[3] * zh.y;
    
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolve L2 with zonal harmonics
// zh.x = L0 scale, zh.y = L1 scale, zh.z = L2 scale
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l2_convolve_zh(sh: SH_L2, zh: vec3<f32>) -> SH_L2 {
    var result: SH_L2;
    
    // L0
    result.c[0] = sh.c[0] * zh.x;
    
    // L1
    result.c[1] = sh.c[1] * zh.y;
    result.c[2] = sh.c[2] * zh.y;
    result.c[3] = sh.c[3] * zh.y;
    
    // L2
    result.c[4] = sh.c[4] * zh.z;
    result.c[5] = sh.c[5] * zh.z;
    result.c[6] = sh.c[6] * zh.z;
    result.c[7] = sh.c[7] * zh.z;
    result.c[8] = sh.c[8] * zh.z;
    
    return result;
}

fn sh_l2_rgb_convolve_zh(sh: SH_L2_RGB, zh: vec3<f32>) -> SH_L2_RGB {
    var result: SH_L2_RGB;
    
    // L0
    result.c[0] = sh.c[0] * zh.x;
    
    // L1
    result.c[1] = sh.c[1] * zh.y;
    result.c[2] = sh.c[2] * zh.y;
    result.c[3] = sh.c[3] * zh.y;
    
    // L2
    result.c[4] = sh.c[4] * zh.z;
    result.c[5] = sh.c[5] * zh.z;
    result.c[6] = sh.c[6] * zh.z;
    result.c[7] = sh.c[7] * zh.z;
    result.c[8] = sh.c[8] * zh.z;
    
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolve with cosine lobe (for converting radiance to irradiance)
// See Ramamoorthi & Hanrahan 2001
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_convolve_cosine_lobe(sh: SH_L1) -> SH_L1 {
    return sh_l1_convolve_zh(sh, vec2<f32>(SH_COSINE_A0, SH_COSINE_A1));
}

fn sh_l1_rgb_convolve_cosine_lobe(sh: SH_L1_RGB) -> SH_L1_RGB {
    return sh_l1_rgb_convolve_zh(sh, vec2<f32>(SH_COSINE_A0, SH_COSINE_A1));
}

fn sh_l2_convolve_cosine_lobe(sh: SH_L2) -> SH_L2 {
    return sh_l2_convolve_zh(sh, vec3<f32>(SH_COSINE_A0, SH_COSINE_A1, SH_COSINE_A2));
}

fn sh_l2_rgb_convolve_cosine_lobe(sh: SH_L2_RGB) -> SH_L2_RGB {
    return sh_l2_rgb_convolve_zh(sh, vec3<f32>(SH_COSINE_A0, SH_COSINE_A1, SH_COSINE_A2));
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                          IRRADIANCE CALCULATION                           ║
// ║                                                                           ║
// ║  Calculate irradiance at a surface from SH-encoded radiance.              ║
// ║  This is the primary use case for diffuse lighting from probes.           ║
// ║                                                                           ║
// ║  NOTE: Does not include 1/PI for Lambertian BRDF.                         ║
// ║  Usage: diffuse = calculate_irradiance(sh, normal) * albedo / PI          ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

fn sh_l1_calculate_irradiance(sh: SH_L1, normal: vec3<f32>) -> f32 {
    let convolved = sh_l1_convolve_cosine_lobe(sh);
    return sh_l1_evaluate(convolved, normal);
}

fn sh_l1_rgb_calculate_irradiance(sh: SH_L1_RGB, normal: vec3<f32>) -> vec3<f32> {
    let convolved = sh_l1_rgb_convolve_cosine_lobe(sh);
    return sh_l1_rgb_evaluate(convolved, normal);
}

fn sh_l2_calculate_irradiance(sh: SH_L2, normal: vec3<f32>) -> f32 {
    let convolved = sh_l2_convolve_cosine_lobe(sh);
    return sh_l2_evaluate(convolved, normal);
}

fn sh_l2_rgb_calculate_irradiance(sh: SH_L2_RGB, normal: vec3<f32>) -> vec3<f32> {
    let convolved = sh_l2_rgb_convolve_cosine_lobe(sh);
    return sh_l2_rgb_evaluate(convolved, normal);
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                    ADVANCED IRRADIANCE CALCULATIONS                       ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Geometrics non-linear fit for L1 irradiance
// See Graham Hazel's "Converting SH Radiance to Irradiance"
// Provides better quality than linear evaluation for L1
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_calculate_irradiance_geometrics(sh: SH_L1, normal: vec3<f32>) -> f32 {
    let r0 = max(sh.c[0], 0.00001);
    
    let r1 = 0.5 * vec3<f32>(sh.c[3], sh.c[1], sh.c[2]);
    let len_r1 = max(length(r1), 0.00001);
    
    let q = 0.5 * (1.0 + dot(r1 / len_r1, normal));
    
    let p = 1.0 + 2.0 * len_r1 / r0;
    let a = (1.0 - len_r1 / r0) / (1.0 + len_r1 / r0);
    
    return r0 * (a + (1.0 - a) * (p + 1.0) * pow(abs(q), p));
}

fn sh_l1_rgb_calculate_irradiance_geometrics(sh: SH_L1_RGB, normal: vec3<f32>) -> vec3<f32> {
    let sh_r = sh_l1_create(sh.c[0].x, sh.c[1].x, sh.c[2].x, sh.c[3].x);
    let sh_g = sh_l1_create(sh.c[0].y, sh.c[1].y, sh.c[2].y, sh.c[3].y);
    let sh_b = sh_l1_create(sh.c[0].z, sh.c[1].z, sh.c[2].z, sh.c[3].z);
    
    return vec3<f32>(
        sh_l1_calculate_irradiance_geometrics(sh_r, normal),
        sh_l1_calculate_irradiance_geometrics(sh_g, normal),
        sh_l1_calculate_irradiance_geometrics(sh_b, normal)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ZH3 Hallucination for L1 irradiance
// See Roughton et al. "ZH3: Quadratic Zonal Harmonics"
// Hallucinates L2 zonal harmonics from L1 for improved quality
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_calculate_irradiance_zh3_hallucinate(sh: SH_L1, normal: vec3<f32>) -> f32 {
    let zonal_axis = normalize(vec3<f32>(sh.c[3], sh.c[1], sh.c[2]));
    
    let ratio = abs(dot(vec3<f32>(sh.c[3], sh.c[1], sh.c[2]), zonal_axis)) / sh.c[0];
    
    let zonal_l2_coeff = sh.c[0] * (0.08 * ratio + 0.6 * ratio * ratio);
    
    let f_z = dot(zonal_axis, normal);
    let zh_dir = sqrt(5.0 / (16.0 * SH_PI)) * (3.0 * f_z * f_z - 1.0);
    
    let base_irradiance = sh_l1_calculate_irradiance(sh, normal);
    
    return base_irradiance + ((SH_PI * 0.25) * zonal_l2_coeff * zh_dir);
}

fn sh_l1_rgb_calculate_irradiance_zh3_hallucinate(sh: SH_L1_RGB, normal: vec3<f32>) -> vec3<f32> {
    let lum_coefficients = vec3<f32>(0.2126, 0.7152, 0.0722);
    let zonal_axis = normalize(vec3<f32>(
        dot(sh.c[3], lum_coefficients),
        dot(sh.c[1], lum_coefficients),
        dot(sh.c[2], lum_coefficients)
    ));
    
    var ratio: vec3<f32>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let dir_i = vec3<f32>(sh.c[3][i], sh.c[1][i], sh.c[2][i]);
        ratio[i] = abs(dot(dir_i, zonal_axis)) / sh.c[0][i];
    }
    
    let zonal_l2_coeff = sh.c[0] * (0.08 * ratio + 0.6 * ratio * ratio);
    
    let f_z = dot(zonal_axis, normal);
    let zh_dir = sqrt(5.0 / (16.0 * SH_PI)) * (3.0 * f_z * f_z - 1.0);
    
    let base_irradiance = sh_l1_rgb_calculate_irradiance(sh, normal);
    
    return base_irradiance + ((SH_PI * 0.25) * zonal_l2_coeff * zh_dir);
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                      OPTIMAL LINEAR DIRECTION                             ║
// ║                                                                           ║
// ║  Compute the "dominant" direction of SH-encoded lighting.                 ║
// ║  See Sloan's "Stupid SH Tricks"                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

fn sh_l1_optimal_linear_direction(sh: SH_L1) -> vec3<f32> {
    return normalize(vec3<f32>(sh.c[3], sh.c[1], sh.c[2]));
}

fn sh_l1_rgb_optimal_linear_direction(sh: SH_L1_RGB) -> vec3<f32> {
    var direction = vec3<f32>(0.0);
    for (var i = 0u; i < 3u; i = i + 1u) {
        direction.x = direction.x + sh.c[3][i];
        direction.y = direction.y + sh.c[1][i];
        direction.z = direction.z + sh.c[2][i];
    }
    return normalize(direction);
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                  DIRECTIONAL LIGHT APPROXIMATION                          ║
// ║                                                                           ║
// ║  Extract a directional light that approximates SH-encoded lighting.       ║
// ║  Useful for specular or shadow estimation from probes.                    ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

struct SH_DirectionalLight {
    direction: vec3<f32>,
    intensity: f32,
}

struct SH_DirectionalLightRGB {
    direction: vec3<f32>,
    color: vec3<f32>,
}

fn sh_l1_approximate_directional_light(sh: SH_L1) -> SH_DirectionalLight {
    var result: SH_DirectionalLight;
    result.direction = sh_l1_optimal_linear_direction(sh);
    
    var dir_sh = sh_project_onto_l1(result.direction, 1.0);
    dir_sh.c[0] = 0.0;  // Zero out L0 term
    
    result.intensity = sh_l1_dot(dir_sh, sh) * (867.0 / (316.0 * SH_PI));
    return result;
}

fn sh_l1_rgb_approximate_directional_light(sh: SH_L1_RGB) -> SH_DirectionalLightRGB {
    var result: SH_DirectionalLightRGB;
    result.direction = sh_l1_rgb_optimal_linear_direction(sh);
    
    var dir_sh = sh_project_onto_l1_rgb(result.direction, vec3<f32>(1.0));
    dir_sh.c[0] = vec3<f32>(0.0);  // Zero out L0 term
    
    result.color = sh_l1_rgb_dot(dir_sh, sh) * (867.0 / (316.0 * SH_PI));
    return result;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                         GGX CONVOLUTION                                   ║
// ║                                                                           ║
// ║  Approximate GGX lobe as zonal harmonics for specular evaluation.         ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Approximate GGX as L1 zonal harmonics (fitted curve)
// ─────────────────────────────────────────────────────────────────────────────
fn sh_approximate_ggx_as_l1_zh(ggx_alpha: f32) -> vec2<f32> {
    let l1_scale = 1.66711256633276 / (1.65715038133932 + ggx_alpha);
    return vec2<f32>(1.0, l1_scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// Approximate GGX as L2 zonal harmonics (fitted curve)
// ─────────────────────────────────────────────────────────────────────────────
fn sh_approximate_ggx_as_l2_zh(ggx_alpha: f32) -> vec3<f32> {
    let l1_scale = 1.66711256633276 / (1.65715038133932 + ggx_alpha);
    let l2_scale = 1.56127990596116 / (0.96989757593282 + ggx_alpha) - 0.599972342361123;
    return vec3<f32>(1.0, l1_scale, l2_scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolve SH with GGX lobe
// ─────────────────────────────────────────────────────────────────────────────

fn sh_l1_convolve_ggx(sh: SH_L1, ggx_alpha: f32) -> SH_L1 {
    return sh_l1_convolve_zh(sh, sh_approximate_ggx_as_l1_zh(ggx_alpha));
}

fn sh_l1_rgb_convolve_ggx(sh: SH_L1_RGB, ggx_alpha: f32) -> SH_L1_RGB {
    return sh_l1_rgb_convolve_zh(sh, sh_approximate_ggx_as_l1_zh(ggx_alpha));
}

fn sh_l2_convolve_ggx(sh: SH_L2, ggx_alpha: f32) -> SH_L2 {
    return sh_l2_convolve_zh(sh, sh_approximate_ggx_as_l2_zh(ggx_alpha));
}

fn sh_l2_rgb_convolve_ggx(sh: SH_L2_RGB, ggx_alpha: f32) -> SH_L2_RGB {
    return sh_l2_rgb_convolve_zh(sh, sh_approximate_ggx_as_l2_zh(ggx_alpha));
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                     SPECULAR DIRECTIONAL LIGHT                            ║
// ║                                                                           ║
// ║  Extract directional light parameters for specular evaluation.            ║
// ║  See Yuriy O'Donnell's "Precomputed Global Illumination in Frostbite"     ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

struct SH_SpecularDirLight {
    direction: vec3<f32>,
    intensity: f32,
    modified_sqrt_roughness: f32,
}

struct SH_SpecularDirLightRGB {
    direction: vec3<f32>,
    color: vec3<f32>,
    modified_sqrt_roughness: f32,
}

fn sh_l1_extract_specular_dir_light(sh_radiance: SH_L1, sqrt_roughness: f32) -> SH_SpecularDirLight {
    var result: SH_SpecularDirLight;
    
    let avg_l1 = vec3<f32>(sh_radiance.c[3], sh_radiance.c[1], sh_radiance.c[2]) * 0.5;
    let avg_l1_len = length(avg_l1);
    
    result.direction = avg_l1 / avg_l1_len;
    result.intensity = sh_l1_evaluate(sh_radiance, result.direction) * SH_PI;
    result.modified_sqrt_roughness = saturate(sqrt_roughness / sqrt(avg_l1_len));
    
    return result;
}

fn sh_l1_rgb_extract_specular_dir_light(sh_radiance: SH_L1_RGB, sqrt_roughness: f32) -> SH_SpecularDirLightRGB {
    var result: SH_SpecularDirLightRGB;
    
    let avg_l1 = vec3<f32>(
        dot(sh_radiance.c[3] / sh_radiance.c[0], vec3<f32>(0.333)),
        dot(sh_radiance.c[1] / sh_radiance.c[0], vec3<f32>(0.333)),
        dot(sh_radiance.c[2] / sh_radiance.c[0], vec3<f32>(0.333))
    ) * 0.5;
    let avg_l1_len = length(avg_l1);
    
    result.direction = avg_l1 / avg_l1_len;
    result.color = sh_l1_rgb_evaluate(sh_radiance, result.direction) * SH_PI;
    result.modified_sqrt_roughness = saturate(sqrt_roughness / sqrt(avg_l1_len));
    
    return result;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                           SH ROTATION                                     ║
// ║                                                                           ║
// ║  Rotate SH coefficients by a rotation matrix.                             ║
// ║  Adapted from DirectX::XMSHRotate (originally by Peter-Pike Sloan)        ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// Rotate L1 SH coefficients
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l1_rotate(sh: SH_L1, rotation: mat3x3<f32>) -> SH_L1 {
    var result: SH_L1;
    
    // L0 is invariant under rotation
    result.c[0] = sh.c[0];
    
    // L1 transforms like a direction vector
    let dir = vec3<f32>(sh.c[3], sh.c[1], sh.c[2]);
    let rotated_dir = rotation * dir;
    result.c[3] = rotated_dir.x;
    result.c[1] = rotated_dir.y;
    result.c[2] = rotated_dir.z;
    
    return result;
}

fn sh_l1_rgb_rotate(sh: SH_L1_RGB, rotation: mat3x3<f32>) -> SH_L1_RGB {
    var result: SH_L1_RGB;
    
    // L0 is invariant under rotation
    result.c[0] = sh.c[0];
    
    // L1 transforms like a direction vector (per channel)
    for (var i = 0u; i < 3u; i = i + 1u) {
        let dir = vec3<f32>(sh.c[3][i], sh.c[1][i], sh.c[2][i]);
        let rotated_dir = rotation * dir;
        result.c[3][i] = rotated_dir.x;
        result.c[1][i] = rotated_dir.y;
        result.c[2][i] = rotated_dir.z;
    }
    
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotate L2 SH coefficients
// Uses adapted DirectX SH rotation with adjusted basis vector conventions
// ─────────────────────────────────────────────────────────────────────────────
fn sh_l2_rotate(sh: SH_L2, rotation: mat3x3<f32>) -> SH_L2 {
    // Basis vector adjustment (DXSH uses slightly different conventions)
    let r00 = rotation[0][0];
    let r10 = rotation[0][1];
    let r20 = -rotation[0][2];
    
    let r01 = rotation[1][0];
    let r11 = rotation[1][1];
    let r21 = -rotation[1][2];
    
    let r02 = -rotation[2][0];
    let r12 = -rotation[2][1];
    let r22 = rotation[2][2];
    
    var result: SH_L2;
    
    // L0 is invariant
    result.c[0] = sh.c[0];
    
    // L1 rotation
    result.c[1] = r11 * sh.c[1] - r12 * sh.c[2] + r10 * sh.c[3];
    result.c[2] = -r21 * sh.c[1] + r22 * sh.c[2] - r20 * sh.c[3];
    result.c[3] = r01 * sh.c[1] - r02 * sh.c[2] + r00 * sh.c[3];
    
    // L2 rotation matrix coefficients
    let t41 = r01 * r00;
    let t43 = r11 * r10;
    let t48 = r11 * r12;
    let t50 = r01 * r02;
    let t55 = r02 * r02;
    let t57 = r22 * r22;
    let t58 = r12 * r12;
    let t61 = r00 * r02;
    let t63 = r10 * r12;
    let t68 = r10 * r10;
    let t70 = r01 * r01;
    let t72 = r11 * r11;
    let t74 = r00 * r00;
    let t76 = r21 * r21;
    let t78 = r20 * r20;
    
    let v173: f32 = 1.732050808;
    let v577: f32 = 0.5773502693;
    let v115: f32 = 1.154700539;
    let v288: f32 = 0.2886751347;
    let v866: f32 = 0.8660254040;
    
    var r: array<f32, 25>;
    r[0] = r11 * r00 + r01 * r10;
    r[1] = -r01 * r12 - r11 * r02;
    r[2] = v173 * r02 * r12;
    r[3] = -r10 * r02 - r00 * r12;
    r[4] = r00 * r10 - r01 * r11;
    r[5] = -r11 * r20 - r21 * r10;
    r[6] = r11 * r22 + r21 * r12;
    r[7] = -v173 * r22 * r12;
    r[8] = r20 * r12 + r10 * r22;
    r[9] = -r10 * r20 + r11 * r21;
    r[10] = -v577 * (t41 + t43) + v115 * r21 * r20;
    r[11] = v577 * (t48 + t50) - v115 * r21 * r22;
    r[12] = -0.5 * (t55 + t58) + t57;
    r[13] = v577 * (t61 + t63) - v115 * r20 * r22;
    r[14] = v288 * (t70 - t68 + t72 - t74) - v577 * (t76 - t78);
    r[15] = -r01 * r20 - r21 * r00;
    r[16] = r01 * r22 + r21 * r02;
    r[17] = -v173 * r22 * r02;
    r[18] = r00 * r22 + r20 * r02;
    r[19] = -r00 * r20 + r01 * r21;
    r[20] = t41 - t43;
    r[21] = -t50 + t48;
    r[22] = v866 * (t55 - t58);
    r[23] = t63 - t61;
    r[24] = 0.5 * (t74 - t68 - t70 + t72);
    
    // Apply L2 rotation
    for (var i = 0u; i < 5u; i = i + 1u) {
        let base = i * 5u;
        result.c[4u + i] = r[base + 0u] * sh.c[4] + r[base + 1u] * sh.c[5] +
                          r[base + 2u] * sh.c[6] + r[base + 3u] * sh.c[7] +
                          r[base + 4u] * sh.c[8];
    }
    
    return result;
}

fn sh_l2_rgb_rotate(sh: SH_L2_RGB, rotation: mat3x3<f32>) -> SH_L2_RGB {
    // Basis vector adjustment
    let r00 = rotation[0][0];
    let r10 = rotation[0][1];
    let r20 = -rotation[0][2];
    
    let r01 = rotation[1][0];
    let r11 = rotation[1][1];
    let r21 = -rotation[1][2];
    
    let r02 = -rotation[2][0];
    let r12 = -rotation[2][1];
    let r22 = rotation[2][2];
    
    var result: SH_L2_RGB;
    
    // L0 is invariant
    result.c[0] = sh.c[0];
    
    // L1 rotation
    result.c[1] = r11 * sh.c[1] - r12 * sh.c[2] + r10 * sh.c[3];
    result.c[2] = -r21 * sh.c[1] + r22 * sh.c[2] - r20 * sh.c[3];
    result.c[3] = r01 * sh.c[1] - r02 * sh.c[2] + r00 * sh.c[3];
    
    // L2 rotation coefficients
    let t41 = r01 * r00;
    let t43 = r11 * r10;
    let t48 = r11 * r12;
    let t50 = r01 * r02;
    let t55 = r02 * r02;
    let t57 = r22 * r22;
    let t58 = r12 * r12;
    let t61 = r00 * r02;
    let t63 = r10 * r12;
    let t68 = r10 * r10;
    let t70 = r01 * r01;
    let t72 = r11 * r11;
    let t74 = r00 * r00;
    let t76 = r21 * r21;
    let t78 = r20 * r20;
    
    let v173: f32 = 1.732050808;
    let v577: f32 = 0.5773502693;
    let v115: f32 = 1.154700539;
    let v288: f32 = 0.2886751347;
    let v866: f32 = 0.8660254040;
    
    var r: array<f32, 25>;
    r[0] = r11 * r00 + r01 * r10;
    r[1] = -r01 * r12 - r11 * r02;
    r[2] = v173 * r02 * r12;
    r[3] = -r10 * r02 - r00 * r12;
    r[4] = r00 * r10 - r01 * r11;
    r[5] = -r11 * r20 - r21 * r10;
    r[6] = r11 * r22 + r21 * r12;
    r[7] = -v173 * r22 * r12;
    r[8] = r20 * r12 + r10 * r22;
    r[9] = -r10 * r20 + r11 * r21;
    r[10] = -v577 * (t41 + t43) + v115 * r21 * r20;
    r[11] = v577 * (t48 + t50) - v115 * r21 * r22;
    r[12] = -0.5 * (t55 + t58) + t57;
    r[13] = v577 * (t61 + t63) - v115 * r20 * r22;
    r[14] = v288 * (t70 - t68 + t72 - t74) - v577 * (t76 - t78);
    r[15] = -r01 * r20 - r21 * r00;
    r[16] = r01 * r22 + r21 * r02;
    r[17] = -v173 * r22 * r02;
    r[18] = r00 * r22 + r20 * r02;
    r[19] = -r00 * r20 + r01 * r21;
    r[20] = t41 - t43;
    r[21] = -t50 + t48;
    r[22] = v866 * (t55 - t58);
    r[23] = t63 - t61;
    r[24] = 0.5 * (t74 - t68 - t70 + t72);
    
    // Apply L2 rotation
    for (var i = 0u; i < 5u; i = i + 1u) {
        let base = i * 5u;
        result.c[4u + i] = r[base + 0u] * sh.c[4] + r[base + 1u] * sh.c[5] +
                          r[base + 2u] * sh.c[6] + r[base + 3u] * sh.c[7] +
                          r[base + 4u] * sh.c[8];
    }
    
    return result;
}

// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                              REFERENCES                                   ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║                                                                           ║
// ║  [0] Stupid SH Tricks - Peter-Pike Sloan                                  ║
// ║      https://www.ppsloan.org/publications/StupidSH36.pdf                  ║
// ║                                                                           ║
// ║  [1] Converting SH Radiance to Irradiance - Graham Hazel                  ║
// ║      https://grahamhazel.com/blog/2017/12/22/                             ║
// ║      converting-sh-radiance-to-irradiance/                                ║
// ║                                                                           ║
// ║  [2] An Efficient Representation for Irradiance Environment Maps          ║
// ║      Ravi Ramamoorthi and Pat Hanrahan                                    ║
// ║      https://cseweb.ucsd.edu/~ravir/6998/papers/envmap.pdf                ║
// ║                                                                           ║
// ║  [3] SHMath - Chuck Walbourn (originally by Peter-Pike Sloan)             ║
// ║      https://walbourn.github.io/spherical-harmonics-math/                 ║
// ║                                                                           ║
// ║  [4] ZH3: Quadratic Zonal Harmonics                                       ║
// ║      Thomas Roughton, Peter-Pike Sloan, Ari Silvennoinen,                 ║
// ║      Michal Iwanicki, and Peter Shirley                                   ║
// ║      https://torust.me/ZH3.pdf                                            ║
// ║                                                                           ║
// ║  [5] Precomputed Global Illumination in Frostbite - Yuriy O'Donnell       ║
// ║      https://www.ea.com/frostbite/news/                                   ║
// ║      precomputed-global-illumination-in-frostbite                         ║
// ║                                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

