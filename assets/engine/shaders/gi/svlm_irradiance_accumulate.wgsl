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

var<workgroup> sh_c0: array<vec3<f32>, 256>;
var<workgroup> sh_c1: array<vec3<f32>, 256>;
var<workgroup> sh_c2: array<vec3<f32>, 256>;
var<workgroup> sh_c3: array<vec3<f32>, 256>;

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
                sample = sh_l1_rgb_add(
                    sample,
                    sh_project_onto_l1_rgb(
                        ray.ray_direction.xyz,
                        ray.radiance.xyz *
                            (SVLM_SPHERE_AREA / f32(rays_per_probe))
                    )
                );
            }
            ray_index_in_probe += SVLM_SH_ACCUMULATE_WORKGROUP_SIZE;
        }
    }

    sh_c0[local_index] = sample.c[0];
    sh_c1[local_index] = sample.c[1];
    sh_c2[local_index] = sample.c[2];
    sh_c3[local_index] = sample.c[3];

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

    var next_sh: SH_L1_RGB;
    next_sh.c[0] = sh_c0[0];
    next_sh.c[1] = sh_c1[0];
    next_sh.c[2] = sh_c2[0];
    next_sh.c[3] = sh_c3[0];
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
