#include "gi/svlm_common.wgsl"
#include "sh_common.wgsl"

// Projects one ray batch into packed L1 RGB SH. A bake sample is one complete
// spherical ray set per probe; subsequent samples are accumulated as an
// unbiased running mean before being packed to f16.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;
@group(1) @binding(2) var<storage, read> ray_data: SVLMProbeRayDataBufferReadOnlyHeader;
@group(1) @binding(3) var<storage, read_write> irradiance_probes: array<u32>;

const SVLM_SH_ACCUMULATE_WORKGROUP_SIZE = 256u;
const SVLM_SPHERE_AREA = 12.566370614359172;

// vec3 arrays already have a 16-byte stride in workgroup memory. Reusing each
// vector's padding lane keeps the full reduction at the WebGPU 16 KiB minimum
// while carrying the backface count without another shared allocation.
var<workgroup> sh_c0: array<vec4<f32>, 256>;
var<workgroup> sh_c1: array<vec4<f32>, 256>;
var<workgroup> sh_c2: array<vec4<f32>, 256>;
var<workgroup> sh_c3: array<vec4<f32>, 256>;

fn svlm_read_probe_sh(probe_index: u32) -> SH_L1_RGB {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    var packed: SH_L1_RGB_Packed;
    for (var i = 0u; i < SVLM_SH_WORDS_PER_PROBE; i = i + 1u) {
        packed.data[i] = irradiance_probes[base + i];
    }
    return sh_l1_rgb_unpack(packed);
}

fn svlm_write_probe_sh(probe_index: u32, sh: SH_L1_RGB) {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    let packed = sh_l1_rgb_pack(sh);
    for (var i = 0u; i < SVLM_SH_WORDS_PER_PROBE; i = i + 1u) {
        irradiance_probes[base + i] = packed.data[i];
    }
}

fn svlm_write_invalid_probe(probe_index: u32) {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    irradiance_probes[base] = SVLM_INVALID_PROBE_WORD_0;
    irradiance_probes[base + 1u] = SVLM_INVALID_PROBE_WORD_1_VALUE;
    irradiance_probes[base + 2u] = 0u;
    irradiance_probes[base + 3u] = 0u;
    irradiance_probes[base + 4u] = 0u;
    irradiance_probes[base + 5u] = 0u;
}

fn svlm_probe_was_invalid(probe_index: u32) -> bool {
    let base = probe_index * SVLM_SH_WORDS_PER_PROBE;
    return
        irradiance_probes[base] == SVLM_INVALID_PROBE_WORD_0 &&
        (irradiance_probes[base + 1u] &
            SVLM_INVALID_PROBE_WORD_1_MASK) ==
            SVLM_INVALID_PROBE_WORD_1_VALUE;
}

@compute @workgroup_size(256, 1, 1)
fn cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let probe_slot = workgroup_id.x;
    let rays_per_probe = max(
        1u,
        u32(svlm_params.irradiance_rays_per_probe)
    );
    let probes_per_batch = max(
        1u,
        u32(svlm_params.irradiance_probes_per_batch)
    );
    let sample_count = max(
        1u,
        u32(svlm_params.irradiance_sample_count)
    );
    let sample_index = atomicLoad(
        &svlm_counters.irradiance_sample_index
    );
    let probe_count = atomicLoad(&svlm_counters.probe_count);
    let cursor = min(
        atomicLoad(&svlm_counters.irradiance_cursor),
        probe_count
    );
    let active_probe_count = min(probes_per_batch, probe_count - cursor);
    let probe_index = cursor + probe_slot;
    let valid_probe =
        sample_index < sample_count &&
        probe_slot < active_probe_count &&
        probe_index * SVLM_SH_WORDS_PER_PROBE +
            (SVLM_SH_WORDS_PER_PROBE - 1u) <
            arrayLength(&irradiance_probes);

    var sample = sh_l1_rgb_zero();
    var backface_count = 0u;
    if (valid_probe) {
        var ray_index_in_probe = local_index;
        loop {
            if (ray_index_in_probe >= rays_per_probe) {
                break;
            }
            let ray_index =
                probe_slot * rays_per_probe + ray_index_in_probe;
            if (
                ray_index < ray_data.header.active_ray_count &&
                ray_index < arrayLength(&ray_data.rays)
            ) {
                let ray = ray_data.rays[ray_index];
                backface_count += select(
                    0u,
                    1u,
                    ray.state_u32.w != INVALID_IDX &&
                        ray.hit_payload_t.w < 0.0
                );
                sample = sh_l1_rgb_add(
                    sample,
                    sh_project_onto_l1_rgb(
                        ray.ray_direction.xyz,
                        svlm_unpack_ray_radiance(ray.radiance) *
                            (SVLM_SPHERE_AREA / f32(rays_per_probe))
                    )
                );
            }
            ray_index_in_probe += SVLM_SH_ACCUMULATE_WORKGROUP_SIZE;
        }
    }

    sh_c0[local_index] = vec4<f32>(sample.c[0], f32(backface_count));
    sh_c1[local_index] = vec4<f32>(sample.c[1], 0.0);
    sh_c2[local_index] = vec4<f32>(sample.c[2], 0.0);
    sh_c3[local_index] = vec4<f32>(sample.c[3], 0.0);

    // A full workgroup reduction is deliberately used here. WGSL does not
    // define a mapping from local invocation indices to subgroup IDs, so
    // deriving subgroup slots from local_index would corrupt probe partials on
    // implementations that choose a different mapping.
    var stride = SVLM_SH_ACCUMULATE_WORKGROUP_SIZE / 2u;
    loop {
        workgroupBarrier();
        if (local_index < stride) {
            sh_c0[local_index] += sh_c0[local_index + stride];
            sh_c1[local_index] += sh_c1[local_index + stride];
            sh_c2[local_index] += sh_c2[local_index + stride];
            sh_c3[local_index] += sh_c3[local_index + stride];
        }
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (local_index != 0u || !valid_probe) {
        return;
    }

    // Consider a probe valid if it doesn't have too many backface hits.
    // Keeping it as a valid black probe creates whole dark rows after
    // trilinear interpolation, so preserve a zero-valued validity marker
    // instead. This matches the established DDGI classification threshold.
    if (
        sh_c0[0].w > f32(rays_per_probe) * 0.25 ||
        (sample_index > 0u && svlm_probe_was_invalid(probe_index))
    ) {
        svlm_write_invalid_probe(probe_index);
        atomicAdd(
            &svlm_counters.irradiance_completed_probe_samples,
            1u
        );
        return;
    }

    var next_sh: SH_L1_RGB;
    next_sh.c[0] = sh_c0[0].xyz;
    next_sh.c[1] = sh_c1[0].xyz;
    next_sh.c[2] = sh_c2[0].xyz;
    next_sh.c[3] = sh_c3[0].xyz;
    if (sample_index > 0u) {
        next_sh = sh_l1_rgb_lerp(
            svlm_read_probe_sh(probe_index),
            next_sh,
            1.0 / f32(sample_index + 1u)
        );
    }

    svlm_write_probe_sh(probe_index, next_sh);
    atomicAdd(
        &svlm_counters.irradiance_completed_probe_samples,
        1u
    );
}
