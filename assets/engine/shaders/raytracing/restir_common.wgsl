const num_ris_samples = 2u;
const num_spatial_samples = 3u;
const num_max_samples = 8u;
const spatial_radius = 20.0;

struct GIReservoir {
    selected_index: u32,
    weight_sum: f32,
    m: u32,                      // Number of samples seen
    w: f32,                      // Final weight for selected sample
};

struct GISample {
    radiance_and_target_pdf: vec4<f32>,         // Total radiance contribution (direct + indirect) + Target PDF for this sample
    direction_and_source_pdf: vec4<f32>,        // Next bounce direction + Source PDF used to generate this sample
};

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
    rng_state: ptr<function, u32>
) {
    (*reservoir).weight_sum += weight;
    (*reservoir).m += 1u;
    
    *rng_state = random_seed(*rng_state);
    let xi = rand_float(*rng_state);
    if (xi * (*reservoir).weight_sum < weight) {
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

// Compute target PDF (p-hat) for a GI sample
// p-hat = luminance(BRDF * radiance)
fn compute_gi_target_pdf(
    sample_radiance: vec3<f32>,
    brdf_value: vec3<f32>
) -> f32 {
    let contribution = sample_radiance * brdf_value;
    let luminance = contribution.x * 0.2126 + contribution.y * 0.7152 + contribution.z * 0.0722;
    return max(luminance, 0.0);
}
