#include "gi/svlm_common.wgsl"

// Advances the GPU-owned progressive bake cursor after a complete trace/shade/
// accumulate batch. Reaching the final sample publishes a sticky ready bit for
// stats readback and future runtime consumers.

@group(1) @binding(0) var<storage, read_write> svlm_params: SVLMParams;
@group(1) @binding(1) var<storage, read_write> svlm_counters: SVLMCounters;

const SVLM_IRRADIANCE_STATUS_COMPLETE = 1u << 0u;

@compute @workgroup_size(1, 1, 1)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) {
        return;
    }

    let target_sample_count = max(
        1u,
        u32(svlm_params.irradiance_sample_count)
    );
    let sample_index = atomicLoad(
        &svlm_counters.irradiance_sample_index
    );
    let probe_count = atomicLoad(&svlm_counters.probe_count);
    let required_probe_samples = probe_count * target_sample_count;
    let completed_probe_samples = atomicLoad(
        &svlm_counters.irradiance_completed_probe_samples
    );
    if (sample_index >= target_sample_count) {
        if (
            probe_count > 0u &&
            completed_probe_samples >= required_probe_samples
        ) {
            atomicOr(
                &svlm_counters.irradiance_status,
                SVLM_IRRADIANCE_STATUS_COMPLETE
            );
        }
        return;
    }

    if (probe_count == 0u) {
        // A zero-probe hierarchy is not a valid baked payload. This most often
        // means the bake was requested before scene acceleration data existed.
        return;
    }

    let cursor = min(
        atomicLoad(&svlm_counters.irradiance_cursor),
        probe_count
    );
    let probes_per_batch = max(
        1u,
        u32(svlm_params.irradiance_probes_per_batch)
    );
    let next_cursor = min(cursor + probes_per_batch, probe_count);
    if (next_cursor < probe_count) {
        atomicStore(&svlm_counters.irradiance_cursor, next_cursor);
        return;
    }

    let next_sample_index = sample_index + 1u;
    atomicStore(&svlm_counters.irradiance_cursor, 0u);
    atomicStore(
        &svlm_counters.irradiance_sample_index,
        next_sample_index
    );
    if (
        next_sample_index >= target_sample_count &&
        atomicLoad(
            &svlm_counters.irradiance_completed_probe_samples
        ) >= required_probe_samples
    ) {
        atomicOr(
            &svlm_counters.irradiance_status,
            SVLM_IRRADIANCE_STATUS_COMPLETE
        );
    }
}
