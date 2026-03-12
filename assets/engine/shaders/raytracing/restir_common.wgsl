const num_init_ris_samples = 4u;

struct GIReservoir {
    selected_index: u32,
    weight_sum: f32,
    m: u32,                      // Number of samples seen
    w: f32,                      // Final weight for selected sample
};

struct GISampleCandidate {
    direction_and_source_pdf: vec4<f32>,
    radiance_and_target_pdf: vec4<f32>,
    lobe_type: u32,
    padding: u32,
    padding2: u32,
    padding3: u32,
};

struct GIReservoirSample {
    visible_position_source_pdf: vec4<f32>,
    sample_position: vec4<f32>,
    sample_normal_target_pdf: vec4<f32>,
    outgoing_radiance_direct: vec4<f32>,
    outgoing_radiance_indirect_diffuse: vec4<f32>,
    outgoing_radiance_indirect_specular: vec4<f32>,
};

struct GIReservoirData {
    reservoir: GIReservoir,
    sample: GIReservoirSample,
};

fn create_empty() -> GIReservoirData {
    var empty: GIReservoirData;
    empty.reservoir = gi_reservoir_init();
    empty.sample.visible_position_source_pdf = vec4<f32>(0.0);
    empty.sample.sample_position = vec4<f32>(0.0);
    empty.sample.sample_normal_target_pdf = vec4<f32>(0.0);
    empty.sample.outgoing_radiance_direct = vec4<f32>(0.0);
    empty.sample.outgoing_radiance_indirect_diffuse = vec4<f32>(0.0);
    empty.sample.outgoing_radiance_indirect_specular = vec4<f32>(0.0);
    return empty;
}

fn gi_reservoir_init() -> GIReservoir {
    var reservoir: GIReservoir;
    reservoir.selected_index = 0u;
    reservoir.weight_sum = 0.0;
    reservoir.m = 0u;
    reservoir.w = 0.0;
    return reservoir;
}

// Update reservoir with a new GI sample candidate
fn gi_reservoir_update(
    reservoir: ptr<function, GIReservoir>,
    candidate_index: u32,
    weight: f32,
    rng_state: ptr<function, u32>,
    sample_clamp_threshold: u32
) {
    (*reservoir).weight_sum += weight;
    (*reservoir).m += 1u;
    (*reservoir).m = min((*reservoir).m, sample_clamp_threshold);
    
    *rng_state = random_seed(*rng_state);
    let xi = rand_float(*rng_state);
    if (xi * (*reservoir).weight_sum < weight) {
        (*reservoir).selected_index = candidate_index;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Update reservoir with explicit random value (for blue noise sampling)
// Takes a pre-computed random value in [0, 1) instead of using RNG state
// ─────────────────────────────────────────────────────────────────────────────
fn gi_reservoir_update_with_rand(
    reservoir: ptr<function, GIReservoir>,
    candidate_index: u32,
    weight: f32,
    xi: f32,
    sample_clamp_threshold: u32
) {
    (*reservoir).weight_sum += weight;
    (*reservoir).m += 1u;
    (*reservoir).m = min((*reservoir).m, sample_clamp_threshold);
    
    if (xi * (*reservoir).weight_sum < weight) {
        (*reservoir).selected_index = candidate_index;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge a source reservoir into a target reservoir
// This properly accumulates the M value from the source reservoir, which is
// critical for unbiased ReSTIR. When merging reservoir B into A:
//   - The combined weight_sum = A.weight_sum + (B.w * B.m * p_hat_at_A)
//   - The combined M = A.m + B.m
// ─────────────────────────────────────────────────────────────────────────────
fn gi_reservoir_merge(
    reservoir: ptr<function, GIReservoir>,
    candidate_index: u32,
    source_reservoir: GIReservoir,
    target_pdf: f32,
    xi: f32,
    sample_clamp_threshold: u32
) {
    if (source_reservoir.m == 0u) {
        return;
    }

    let old_m = (*reservoir).m;

    // If this merge would push us past the clamp, scale the contribution so
    // weight_sum remains consistent with the (clamped) effective M.
    let remaining_m = sample_clamp_threshold - old_m;
    let effective_m = min(source_reservoir.m, remaining_m);
    if (effective_m == 0u) {
        return;
    }

    let m_scale = f32(effective_m) / f32(source_reservoir.m);

    // The contribution weight for the source reservoir's sample when evaluated
    // at the target location: w_i = W_s * M_s * p_hat(x_s)
    let contribution_weight = source_reservoir.w * f32(source_reservoir.m) * target_pdf;
    let scaled_contribution_weight = contribution_weight * m_scale;
    
    (*reservoir).weight_sum += scaled_contribution_weight;
    (*reservoir).m += effective_m;
    
    if (xi * (*reservoir).weight_sum < scaled_contribution_weight) {
        (*reservoir).selected_index = candidate_index;
    }
}

// Finalize GI reservoir and compute final weight
fn gi_reservoir_finalize(
    reservoir: ptr<function, GIReservoir>,
    selected_target_pdf: f32
) {
    let contributes = (*reservoir).m > 0u && selected_target_pdf > 0.0;
    let unclamped_weight = (*reservoir).weight_sum / (f32((*reservoir).m) * max(selected_target_pdf, 0.0001));
    
    // Reasonable clamping to handle extreme variance
    // With proper MIS, variance should be much lower, but still cap to prevent numerical issues
    let max_weight = 200.0;
    (*reservoir).w = select(
        0.0,
        min(max_weight, unclamped_weight),
        contributes
    );
}

// Full Jacobian determinant for spatial reuse (Eq. 11 in ReSTIR GI paper).
// Converts the source pixel's solid angle measure at x_v^q to the target pixel's
// solid angle measure at x_v^r for a shared sample point x_s.
fn compute_restir_gi_jacobian(
    sample_point_normal: vec3<f32>,
    source_visible_position: vec3<f32>,
    target_visible_position: vec3<f32>,
    sample_position: vec3<f32>
) -> f32 {
    let v_q = source_visible_position - sample_position;
    let v_r = target_visible_position - sample_position;

    let dist2_q = max(dot(v_q, v_q), 1e-6);
    let dist2_r = max(dot(v_r, v_r), 1e-6);

    let dir_q = v_q * inverseSqrt(dist2_q);
    let dir_r = v_r * inverseSqrt(dist2_r);

    let cos_q = max(dot(sample_point_normal, dir_q), 0.0);
    let cos_r = max(dot(sample_point_normal, dir_r), 0.0);

    // |J_{q->r}| = (|cos(phi_2^r)| / |cos(phi_2^q)|) * (||x1^q - x2^q||^2 / ||x1^r - x2^q||^2)
    return (cos_r / max(cos_q, 1e-6)) * (dist2_q / dist2_r);
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute target PDF with Jacobian adjustment for spatial reuse
// When reusing a sample from a different surface, apply Jacobian to account
// for the change in solid angle measure between surfaces.
// ─────────────────────────────────────────────────────────────────────────────
fn compute_reuse_target_pdf(
    sample: GIReservoirSample,
    target_visible_position: vec3<f32>
) -> f32 {
    let source_visible_position = sample.visible_position_source_pdf.xyz;
    let sample_position = sample.sample_position.xyz;
    let sample_normal = safe_normalize(sample.sample_normal_target_pdf.xyz);

    let jacobian = compute_restir_gi_jacobian(
        sample_normal,
        source_visible_position,
        target_visible_position,
        sample_position
    );

    let clamped_jacobian = clamp(jacobian, 1e-6, 10.0);
    return sample.sample_normal_target_pdf.w * clamped_jacobian;
}

// ─────────────────────────────────────────────────────────────────────────────
// Temporal radiance similarity check
// NOTE:
// - We compare the reprojected (history) sample's radiance against the current
//   frame's newly traced candidate (candidate_samples[0]) when available.
// - This is an additional disocclusion/lighting-change rejection to avoid
//   dragging stale bright samples across edges or through lighting changes.
// ─────────────────────────────────────────────────────────────────────────────
fn gi_sample_total_radiance(sample: GIReservoirSample) -> vec3<f32> {
    return sample.outgoing_radiance_direct.xyz
        + sample.outgoing_radiance_indirect_diffuse.xyz
        + sample.outgoing_radiance_indirect_specular.xyz;
}
