export const no_cull_rasterizer_config = {
  rasterizer_state: {
    cull_mode: "none",
  },
};

export const one_one_blend_config = {
  color: {
    srcFactor: "one",
    dstFactor: "one",
  },
  alpha: {
    srcFactor: "one",
    dstFactor: "one",
  },
};

export const zero_one_minus_src_blend_config = {
  color: {
    srcFactor: "zero",
    dstFactor: "one-minus-src",
  },
  alpha: {
    srcFactor: "zero",
    dstFactor: "one-minus-src",
  },
};

export const src_alpha_one_minus_src_alpha_blend_config = {
  color: {
    srcFactor: "src-alpha",
    dstFactor: "one-minus-src-alpha",
  },
  alpha: {
    srcFactor: "src-alpha",
    dstFactor: "one-minus-src-alpha",
  },
};

export const rgba32float_format = "rgba32float";
export const rgba16float_format = "rgba16float";
export const rgba32uint_format = "rgba32uint";
export const rgba32sint_format = "rgba32sint";
export const r8unorm_format = "r8unorm";
export const depth32float_format = "depth32float";
export const bgra8unorm_format = "bgra8unorm";
export const rgba8unorm_format = "rgba8unorm";
export const rgba8snorm_format = "rgba8snorm";
export const rgba8uint_format = "rgba8uint";
export const rgba8sint_format = "rgba8sint";
export const rg32float_format = "rg32float";
export const r32float_format = "r32float";
export const r16float_format = "r16float";
export const r32uint_format = "r32uint";
export const rg32uint_format = "rg32uint";
export const r32sint_format = "r32sint";
export const r16uint_format = "r16uint";
export const r16sint_format = "r16sint";

export const load_op_clear = "clear";
export const load_op_load = "load";
export const load_op_dont_care = "dont-care";