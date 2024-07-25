fn reinhard_tonemapping(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    var adjusted_color = color * exposure;
    return adjusted_color / (adjusted_color + vec3<f32>(1.0));
}

fn u2_filmic_tonemapping(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    var adjusted_color = color * exposure;

    let A: f32 = 0.15;
    let B: f32 = 0.50;
    let C: f32 = 0.10;
    let D: f32 = 0.20;
    let E: f32 = 0.02;
    let F: f32 = 0.30;

    return ((adjusted_color * (A * adjusted_color + vec3<f32>(C * B)) + vec3<f32>(D * E)) / (adjusted_color * (A * adjusted_color + vec3<f32>(B)) + vec3<f32>(D * F))) - vec3<f32>(E / F);
}

fn aces_filmic_tonemapping(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

fn aces_tonemapping(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    var adjusted_color = color * exposure;
    adjusted_color = aces_filmic_tonemapping(adjusted_color);
    return adjusted_color;
}