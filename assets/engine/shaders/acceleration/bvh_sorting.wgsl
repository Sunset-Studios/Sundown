#include "common.wgsl"
#include "acceleration_common.wgsl"

////////////////////////////////////////////////////////////////////////////////
// OneSweep (WGSL, logical-warp version when there's no subgroups support)
// Modified and Ported to WGSL (logical warps, no subgroups)
//
// Original: HLSL by Thomas Smith (2024-03-14)
// Based on research by Adinets & Merrill (NVIDIA) - OneSweep (2022)
//
// Note: Requires minimum workgroup storage size of 32KB.
////////////////////////////////////////////////////////////////////////////////

// ==================================
// Tunables / Constants
// ==================================
const PASS_DIM            : u32 = 256u;    // threads in DigitBinningPass workgroup
const G_HIST_DIM          : u32 = 128u;    // threads in GlobalHistogram workgroup
const PART_SIZE           : u32 = 4096u;   // size of a partition tile (KEYS_PER_THREAD * PASS_DIM)

const RADIX               : u32 = 256u;
const RADIX_MASK          : u32 = 255u;
const RADIX_LOG           : u32 = 8u;
const RADIX_PASSES        : u32 = 4u;
const HALF_RADIX          : u32 = RADIX >> 1u;

const SEC_RADIX_START     : u32 = 256u;
const THIRD_RADIX_START   : u32 = 512u;
const FOURTH_RADIX_START  : u32 = 768u;

const KEYS_PER_THREAD     : u32 = 16u;

// decoupled look-back flags
const FLAG_NOT_READY      : u32 = 0u;
const FLAG_REDUCTION      : u32 = 1u;
const FLAG_INCLUSIVE      : u32 = 2u;
const FLAG_MASK           : u32 = 3u;

// ==================================
// Data Structures 
// ==================================
struct Params {
  key_count: u32,
  radix_shift: u32,
  thread_blocks: u32,
  _padding: u32,
};

struct BufU32 {
  data: array<u32>,
};

struct BufA32 {
  data: array<atomic<u32>>,
};

// ==================================
// Bindings
// ==================================
@group(1) @binding(0) var<storage, read_write>  keys_buffer        : BufU32;
@group(1) @binding(1) var<storage, read_write>  scatter_out        : BufU32;
@group(1) @binding(2) var<storage, read_write>  values_buffer      : BufU32;
@group(1) @binding(3) var<storage, read_write>  values_scatter_out : BufU32;
@group(1) @binding(4) var<storage, read_write>  global_historgram  : BufA32;   // atomic
@group(1) @binding(5) var<storage, read_write>  pass_histogram     : BufA32;   // atomic (flags|payload)
@group(1) @binding(6) var<storage, read_write>  tile_indices       : BufA32;   // atomic (one per pass)
@group(1) @binding(7) var<uniform> params : Params;

// ==================================
// Workgroup (shared) memory
// ==================================
// Packed per-tile local histograms: 4 components (x,y,z,w) * RADIX bins.
// Each u32 packs two 16-bit halves: low 16 for lanes <64, high 16 for lanes >=64.
var<workgroup> global_hist_packed : array<atomic<u32>, RADIX * 4u>;
// Per-tile local reduction values
var<workgroup> pass_hist          : array<atomic<u32>, PART_SIZE>;
// Per-warp scan values
var<workgroup> scan               : array<u32, RADIX>;
// Per-digit global base for this tile (computed by decoupled look-back)
var<workgroup> digit_base         : array<u32, RADIX>;

// ==================================
// Helpers Functions
// ==================================
fn extract_digit(key: u32, shift: u32) -> u32 {
  return (key >> shift) & RADIX_MASK;
}

fn extract_packed_index(key: u32, shift: u32) -> u32 {
  return extract_digit(key, shift) >> 1u;
}

fn extract_packed_shift(key: u32, shift: u32) -> u32 {
  let d = extract_digit(key, shift);
  return select(0u, 16u, (d & 1u) != 0u);
}

fn extract_packed_value(word: u32, key: u32, shift: u32) -> u32 {
  let s = extract_packed_shift(key, shift);
  return (word >> s) & 0xFFFFu;
}

fn current_pass() -> u32 {
  return (params.radix_shift >> 3u); // shift / RADIX_LOG
}

fn pass_hist_offset(tile_index: u32) -> u32 {
  return ((current_pass() * params.thread_blocks) + tile_index) << RADIX_LOG; 
}

fn wave_hists_size_ge16(_c: WarpCtx) -> u32 {
  return PASS_DIM / _c.warp_size * RADIX;
}

fn wave_hists_size_lt16(_c: WarpCtx) -> u32 {
  return PART_SIZE;
}

fn subpart_size_ge16(_c: WarpCtx) -> u32
{
    return KEYS_PER_THREAD * _c.warp_size;
}

fn shared_offset_ge16(_c: WarpCtx, local_id: u32) -> u32
{
    return _c.lane_id + _c.warp_id * subpart_size_ge16(_c);
}

fn subpart_size_lt16(_c: WarpCtx, serial_iters: u32) -> u32
{
    return KEYS_PER_THREAD * _c.warp_size * serial_iters;
}

fn shared_offset_lt16(_c: WarpCtx, local_id: u32, serial_iters: u32) -> u32
{
    return _c.lane_id +
        (_c.warp_id / serial_iters * subpart_size_lt16(_c, serial_iters)) +
        (_c.warp_id % serial_iters * _c.warp_size);
}

fn shared_offset(_c: WarpCtx, local_id: u32, serial_iters: u32) -> u32
{
    return select(
        shared_offset_ge16(_c, local_id),
        shared_offset_lt16(_c, local_id, serial_iters),
        _c.warp_size < 16u
    );
}

fn device_offset(_c: WarpCtx, local_id: u32, tile_idx: u32, serial_iters: u32) -> u32
{
    return shared_offset(_c, local_id, serial_iters) + tile_idx * PART_SIZE;
}

fn global_hist_offset(pass_idx: u32) -> u32 {
    return pass_idx * RADIX;
}

// ==================================
// Kernels
// ==================================
// InitOneSweep: clears passHist, globalHist, and tile index counters.
@compute @workgroup_size(256)
fn onesweep_init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;

  // Clear pass_histogram: e_threadBlocks * RADIX * RADIX_PASSES
  let clear_end = params.thread_blocks * RADIX * RADIX_PASSES;
  for (var i: u32 = id; i < clear_end; i += 65536u) {
    atomicStore(&pass_histogram.data[i], 0u);
  }
  // Clear global_historgram: RADIX * RADIX_PASSES
  if (id < RADIX * RADIX_PASSES) {
    atomicStore(&global_historgram.data[id], 0u);
  }
  // Reset per-pass tile indices
  if (id < RADIX_PASSES) {
    atomicStore(&tile_indices.data[id], 0u);
  }
}

// GlobalHistogram: per-tile histogram (4 bytes per key)
@compute @workgroup_size(G_HIST_DIM)
fn onesweep_global_histogram(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>
) {
  let tile_ix = wid.x;

  // zero local hist (packed halves for x,y,z,w)
  for (var j: u32 = lid.x; j < RADIX * 4u; j += G_HIST_DIM) {
    atomicStore(&global_hist_packed[j], 0u);
  }
  workgroupBarrier();

  // choose half by lane range
  let half_inc = select(1u, (1u << 16u), lid.x >= 64u);
  // tile range
  let start = tile_ix * PART_SIZE;
  let end   = select((tile_ix + 1u) * PART_SIZE, params.key_count, tile_ix == (params.thread_blocks - 1u));

  // strided loop over this tile
  for (var idx: u32 = start + lid.x; idx < end; idx += G_HIST_DIM) {
    let t = keys_buffer.data[idx];
    let d0 = extract_digit(t, 0u);
    let d1 = extract_digit(t, 8u);
    let d2 = extract_digit(t, 16u);
    let d3 = extract_digit(t, 24u);

    let o0 = d0 + RADIX * 0u;
    let o1 = d1 + RADIX * 1u;
    let o2 = d2 + RADIX * 2u;
    let o3 = d3 + RADIX * 3u;

    atomicAdd(&global_hist_packed[o0], half_inc);
    atomicAdd(&global_hist_packed[o1], half_inc);
    atomicAdd(&global_hist_packed[o2], half_inc);
    atomicAdd(&global_hist_packed[o3], half_inc);
  }

  workgroupBarrier();

  // reduce packed halves to global hist (atomic adds)
  for (var k: u32 = lid.x; k < RADIX; k += G_HIST_DIM) {
    let px = atomicLoad(&global_hist_packed[k + RADIX * 0u]);
    let py = atomicLoad(&global_hist_packed[k + RADIX * 1u]);
    let pz = atomicLoad(&global_hist_packed[k + RADIX * 2u]);
    let pw = atomicLoad(&global_hist_packed[k + RADIX * 3u]);
    let gx = (px & 0xFFFFu) + (px >> 16u);
    let gy = (py & 0xFFFFu) + (py >> 16u);
    let gz = (pz & 0xFFFFu) + (pz >> 16u);
    let gw = (pw & 0xFFFFu) + (pw >> 16u);

    atomicAdd(&global_historgram.data[k],                      gx);
    atomicAdd(&global_historgram.data[k + SEC_RADIX_START],    gy);
    atomicAdd(&global_historgram.data[k + THIRD_RADIX_START],  gz);
    atomicAdd(&global_historgram.data[k + FOURTH_RADIX_START], gw);
  }
}

// Scan: block-wide version matching HLSL semantics (>=16-lane path),
// using logical warps (32) and circular write into tile 0 plane.
@compute @workgroup_size(RADIX)
fn onesweep_scan(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
#if HAS_SUBGROUPS
  @builtin(subgroup_invocation_id)  sid: u32,
  @builtin(subgroup_size)       ss: u32
#endif
) {
  let l  = lid.x;
#if HAS_SUBGROUPS
  let li = sid;
#else
  let li = lane_id(l, LOGICAL_WARP_SIZE);
#endif

#if HAS_SUBGROUPS
  let warp_ctx = make_warp_ctx(l, li, ss);
#else
  let warp_ctx = make_warp_ctx(l, li, LOGICAL_WARP_SIZE);
#endif

  // Load per-bin counts for THIS PASS from global histogram
  let gh_base = global_hist_offset(current_pass());
  let my_val  = atomicLoad(&global_historgram.data[gh_base + l]);

  // Intra-warp inclusive scan (g_scan += WavePrefixSum(g_scan))
  var incl = warp_scan_exclusive_add_u32(warp_ctx, my_val);
  scan[l] = incl;

  // We assume warp sizes that are generally 16 or larger. Very rare to get warp sizes that are smaller than that.
  //if (warp_ctx.warp_size >= 16u) {
  workgroupBarrier();

  // Prefix over per-warp end elements using first (RADIX / warp_size) threads
  var end_idx: u32 = 0u;
  var end_val: u32 = 0u;
  let in_range_end = l < (RADIX / warp_ctx.warp_size);
  if (in_range_end) {
    end_idx = ((l + 1u) * warp_ctx.warp_size) - 1u;
    end_val = scan[end_idx];
  }
  let end_psum_excl = warp_scan_exclusive_add_u32(warp_ctx, end_val);
  if (in_range_end) {
    scan[end_idx] = end_val + end_psum_excl;
  }
  workgroupBarrier();

  // Circular scatter with previous-warp sum
  let lane_mask = warp_ctx.warp_size - 1u;
  let index = ((li + 1u) & lane_mask) + (l & ~lane_mask);
  let left_val = select(0u, scan[l], li != lane_mask);
  // broadcast previous-warp last element from lane 0 uniformly
  var prev_warp_src: u32 = 0u;
  if (li == 0u && l >= warp_ctx.warp_size) {
    prev_warp_src = scan[l - 1u];
  }
  let prev_warp = warp_broadcast_u32(warp_ctx, prev_warp_src, 0u);
  let write_val = left_val + prev_warp;
  let out_addr = wid.x * params.thread_blocks * RADIX + index;
  atomicStore(&pass_histogram.data[out_addr], (write_val << 2u) | FLAG_INCLUSIVE);
  //}
  
//   if (warp_ctx.warp_size < 16u) {
//     // Fallback hierarchical path for very small wave sizes (<16)
//     let pass_offs = wid.x * params.thread_blocks * RADIX;
//     if (l < warp_ctx.warp_size) {
//       let circular_lane_shift = (li + 1u) & (warp_ctx.warp_size - 1u);
//       let v0 = select(0u, scan[l], circular_lane_shift != 0u);
//       atomicStore(&pass_histogram.data[pass_offs + circular_lane_shift], (v0 << 2u) | FLAG_INCLUSIVE);
//     }
//     workgroupBarrier();

//     let lane_log = countOneBits(warp_ctx.warp_size - 1u);
//     var offset = lane_log;
//     var j = warp_ctx.warp_size;
//     for (var j = warp_ctx.warp_size; j < (RADIX >> 1u); j = j << lane_log) {
//       if (l < (RADIX >> offset)) {
//         let idx2 = ((l + 1u) << offset) - 1u;
//         let v2 = scan[idx2];
//         let p2_excl = warp_scan_exclusive_add_u32(warp_ctx, v2);
//         scan[idx2] = v2 + p2_excl;
//       }
//       workgroupBarrier();

//       let base_idx = ((l >> offset) << offset) - 1u;
//       let base_sum = warp_broadcast_u32(warp_ctx, scan[base_idx], 0u);
//       if ((l & ((j << lane_log) - 1u)) >= j) {
//         if (l < (j << lane_log)) {
//           let addend = select(0u, scan[l - 1u], (l & (j - 1u)) != 0u);
//           atomicStore(&pass_histogram.data[pass_offs + l], ((base_sum + addend) << 2u) | FLAG_INCLUSIVE);
//         } else {
//           if (((l + 1u) & (j - 1u)) != 0u) {
//             scan[l] = scan[l] + base_sum;
//           }
//         }
//       }
//       offset = offset + lane_log;
//     }
//     workgroupBarrier();

//     let index2 = l + j;
//     let base_idx3 = ((index2 >> offset) << offset) - 1u;
//     let base_sum3 = warp_broadcast_u32(warp_ctx, scan[base_idx3], 0u);
//     if (index2 < RADIX) {
//       let addend3 = select(0u, scan[index2 - 1u], (index2 & (j - 1u)) != 0u);
//       atomicStore(&pass_histogram.data[pass_offs + index2], ((base_sum3 + addend3) << 2u) | FLAG_INCLUSIVE);
//     }
//   }
}

// DigitBinningPass: chained scan w/ decoupled look-back, logical warps (32)
@compute @workgroup_size(PASS_DIM)
fn onesweep_digit_binning(
  @builtin(local_invocation_id)  lid: vec3<u32>,
#if HAS_SUBGROUPS
  @builtin(subgroup_invocation_id)  sid: u32,
  @builtin(subgroup_size)       ss: u32
#endif
) {
  let l = lid.x;

#if HAS_SUBGROUPS
  let li = sid;
  let warp_ctx = make_warp_ctx(l, li, ss);
#else
  let li = lane_id(l, LOGICAL_WARP_SIZE);
  let warp_ctx = make_warp_ctx(l, li, LOGICAL_WARP_SIZE);
#endif

  // serial iterations following HLSL: (PASS_DIM / WaveSize + 31) / 32
  let serial_iterations: u32 = (PASS_DIM / warp_ctx.warp_size + 31u) / 32u;

  // determine partition index and clear shared memory
  var tile_idx: u32;
  // We assume warp sizes that are generally 16 or larger. Very rare to get warp sizes that are smaller than that.
  //if (warp_ctx.warp_size > 16u) {
    // clear only wave-hists region
  let hist_area = wave_hists_size_ge16(warp_ctx);
  for (var i: u32 = l; i < hist_area; i += PASS_DIM) {
    atomicStore(&pass_hist[i], 0u);
  }
  if (l == 0u) {
    atomicStore(&pass_hist[PART_SIZE - 1u], atomicAdd(&tile_indices.data[current_pass()], 1u));
  }
  workgroupBarrier();
  tile_idx = atomicLoad(&pass_hist[PART_SIZE - 1u]);
  //}
  
//   if (warp_ctx.warp_size <= 16u) {
//     if (l == 0u) {
//       atomicStore(&pass_hist[0u], atomicAdd(&tile_indices.data[current_pass()], 1u));
//     }
//     workgroupBarrier();
//     tile_idx = atomicLoad(&pass_hist[0u]);
//     workgroupBarrier();
//     // clear full shared pass buffer (bounded by allocation)
//     for (var j: u32 = l; j < PART_SIZE; j += PASS_DIM) {
//       atomicStore(&pass_hist[j], 0u);
//     }
//     workgroupBarrier();
//   }

  // load keys assigned to this workgroup/tile
  var keys: array<u32, KEYS_PER_THREAD>;
  var vals: array<u32, KEYS_PER_THREAD>;
  var offs: array<u32, KEYS_PER_THREAD>;

  let serial_iters: u32 = serial_iterations; // matches HLSL formula, typically 1
  let base = device_offset(warp_ctx, l, tile_idx, serial_iters);

  if (tile_idx < (params.thread_blocks - 1u)) {
    var t: u32 = base;
    for (var j0: u32 = 0u; j0 < KEYS_PER_THREAD; j0 = j0 + 1u) {
      keys[j0] = keys_buffer.data[t];
      vals[j0] = values_buffer.data[t];
      t += warp_ctx.warp_size * serial_iters;
    }
  }
  
  if (tile_idx == (params.thread_blocks - 1u)) {
    // last (possibly partial) tile
    var t2: u32 = base;
    for (var j1: u32 = 0u; j1 < KEYS_PER_THREAD; j1 = j1 + 1u) {
      let in_range = t2 < params.key_count;
      keys[j1] = select(0xffffffffu, keys_buffer.data[t2], in_range);
      vals[j1] = select(0xffffffffu, values_buffer.data[t2], in_range);
      t2 += warp_ctx.warp_size * serial_iters;
    }
  }

  // -------- Tile-exclusive prefix per digit and per-lane base (>=16 lanes) --------
  var exclusive_hist_reduction: u32 = 0u;
  // We assume warp sizes that are generally 16 or larger. Very rare to get warp sizes that are smaller than that.
  //if (warp_ctx.warp_size >= 16u) {
  let wave_parts = (warp_ctx.warp_size + 31u) / 32u;

  // Calculate wave flags for each digit (4 parts)
  for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
    var wave_flags = vec4<u32>(
        select(0u, 0xFFFFFFFFu, warp_ctx.warp_size > 0u),
        select(0u, 0xFFFFFFFFu, warp_ctx.warp_size > 32u),
        select(0u, 0xFFFFFFFFu, warp_ctx.warp_size > 64u),
        select(0u, 0xFFFFFFFFu, warp_ctx.warp_size > 96u)
    );

    // 1. Calculate wave flags for each digit
    for (var j: u32 = 0u; j < RADIX_LOG; j = j + 1u) {
        let t = ((keys[i] >> (j + params.radix_shift)) & 1u) != 0u;
        let ballot = warp_ballot_u32(warp_ctx, t);
        for (var k: u32 = 0u; k < wave_parts; k = k + 1u) {
            let ballot_flag = select(0xFFFFFFFFu, 0u, t) ^ ballot[k];
            wave_flags[k] = wave_flags[k] & ballot_flag;
        }
    }

    // 2. Count bits in wave flags
    var bits: u32 = 0u;
    for (var k: u32 = 0u; k < wave_parts; k = k + 1u) {
        if (warp_ctx.lane_id < k * 32u) {
            bits = bits + countOneBits(wave_flags[k]);
        }
    }

    // 3. Calculate index
    let index = extract_digit(keys[i], params.radix_shift) + (warp_ctx.warp_id * RADIX);
    offs[i] = atomicLoad(&pass_hist[index]) + bits;
   
    workgroupBarrier();

    // 3. Add bits to pass if no bits are set
    if (bits == 0u) {
        for (var k: u32 = 0u; k < wave_parts; k = k + 1u) {
            let add_bits = countOneBits(wave_flags[k]);
            atomicAdd(&pass_hist[index], add_bits);
        }
    }

    workgroupBarrier();
  }

  var hist_reduction: u32 = 0u;
  if (l < RADIX) {
    hist_reduction = atomicLoad(&pass_hist[l]);
    let wave_hist_size = wave_hists_size_ge16(warp_ctx);
    for (var i: u32 = l + RADIX; i < wave_hist_size; i = i + RADIX) {
      let hist_i = atomicLoad(&pass_hist[i]);
      hist_reduction = hist_reduction + hist_i;
      atomicStore(&pass_hist[i], hist_reduction - hist_i);
    }

    if (tile_idx < (params.thread_blocks - 1u)) {
      atomicAdd(&pass_histogram.data[l + pass_hist_offset(tile_idx + 1u)], (FLAG_REDUCTION | (hist_reduction << 2u)));
    }
  }
  hist_reduction = hist_reduction + warp_scan_exclusive_add_u32(warp_ctx, hist_reduction);

  workgroupBarrier();

  // Within-warp inclusive prefix of total per digit then circular scatter
  if (l < RADIX) {
    // circular scatter
    let lane_mask = warp_ctx.warp_size - 1u;
    let dst = ((li + 1u) & lane_mask) + (l & ~lane_mask);
    atomicStore(&pass_hist[dst], hist_reduction);
  }
  workgroupBarrier();

  // prefix at warp boundaries for each digit (uniform call with masking)
  let in_range_boundaries = l < (RADIX / warp_ctx.warp_size);
  var idx0: u32 = 0u;
  var boundary_val: u32 = 0u;
  if (in_range_boundaries) {
    idx0 = l * warp_ctx.warp_size;
    boundary_val = atomicLoad(&pass_hist[idx0]);
  }
  let boundary_excl = warp_scan_exclusive_add_u32(warp_ctx, boundary_val);
  if (in_range_boundaries) {
    atomicStore(&pass_hist[idx0], boundary_excl);
  }
  workgroupBarrier();

  // add lane-1 broadcast to non-zero lanes (uniform call with masking)
  var lane1_prev: u32 = 0u;
  if (li == 1u && l < RADIX) {
    lane1_prev = atomicLoad(&pass_hist[l - 1u]);
  }
  let prev_from_lane1 = warp_broadcast_u32(warp_ctx, lane1_prev, 1u);
  if (l < RADIX && li != 0u) {
    atomicAdd(&pass_hist[l], prev_from_lane1);
  }
  workgroupBarrier();

  // add per-lane base to offsets
  if (l >= warp_ctx.warp_size) {
    let t = warp_ctx.warp_id * RADIX;
    for (var j2: u32 = 0u; j2 < KEYS_PER_THREAD; j2 = j2 + 1u) {
      let d = extract_digit(keys[j2], params.radix_shift);
      offs[j2] = offs[j2] + atomicLoad(&pass_hist[d + t]) + atomicLoad(&pass_hist[d]);
    }
  } else {
    for (var j3: u32 = 0u; j3 < KEYS_PER_THREAD; j3 = j3 + 1u) {
      let d = extract_digit(keys[j3], params.radix_shift);
      offs[j3] = offs[j3] + atomicLoad(&pass_hist[d]);
    }
  }

  if (l < RADIX) {
    exclusive_hist_reduction = atomicLoad(&pass_hist[l]);
  }
  workgroupBarrier();
  //}
  
//   if (warp_ctx.warp_size < 16u) {
//     // -------- WaveGetLaneCount() < 16 path --------
//     let lt_mask = (1u << warp_ctx.lane_id) - 1u;

//     // Per-key waveFlag accumulation and offsets with serial iterations
//     for (var ii: u32 = 0u; ii < KEYS_PER_THREAD; ii = ii + 1u) {
//       var wave_flag: u32 = (1u << warp_ctx.warp_size) - 1u;

//       for (var kb: u32 = 0u; kb < RADIX_LOG; kb = kb + 1u) {
//         let t = ((keys[ii] >> (kb + params.radix_shift)) & 1u) != 0u;
//         let ballot_scalar = warp_ballot_u32(warp_ctx, t).x;
//         let inv_mask = select(0xFFFFFFFFu, 0u, t);
//         wave_flag = wave_flag & (inv_mask ^ ballot_scalar);
//       }

//       let bits = countOneBits(wave_flag & lt_mask);
//       let index = extract_packed_index(keys[ii], params.radix_shift) +
//                   ((warp_ctx.warp_id / serial_iters) * HALF_RADIX);

//       for (var kk: u32 = 0u; kk < serial_iters; kk = kk + 1u) {
//         let is_my_iter = (warp_ctx.warp_id % serial_iters) == kk;
//         if (is_my_iter) {
//           offs[ii] = extract_packed_value(atomicLoad(&pass_hist[index]), keys[ii], params.radix_shift) + bits;
//         }

//         workgroupBarrier();

//         if (is_my_iter && bits == 0u) {
//           let add_bits = countOneBits(wave_flag) << extract_packed_shift(keys[ii], params.radix_shift);
//           atomicAdd(&pass_hist[index], add_bits);
//         }

//         workgroupBarrier();
//       }
//     }

//     // Histogram reduction over HALF_RADIX stripes
//     if (l < HALF_RADIX) {
//       var hist_reduction2: u32 = atomicLoad(&pass_hist[l]);
//       let h_end = wave_hists_size_lt16(warp_ctx);
//       for (var i: u32 = l + HALF_RADIX; i < h_end; i = i + HALF_RADIX) {
//         hist_reduction2 = hist_reduction2 + atomicLoad(&pass_hist[i]);
//         atomicStore(&pass_hist[i], hist_reduction2 - atomicLoad(&pass_hist[i]));
//       }
//       atomicStore(&pass_hist[l], hist_reduction2 + (hist_reduction2 << 16u));

//       if (tile_idx < (params.thread_blocks - 1u)) {
//         atomicAdd(
//             &pass_histogram.data[(l << 1u) + pass_hist_offset(tile_idx + 1u)],
//             (FLAG_REDUCTION | ((hist_reduction2 & 0xFFFFu) << 2u))
//         );
//         atomicAdd(
//             &pass_histogram.data[(l << 1u) + 1u + pass_hist_offset(tile_idx + 1u)],
//             (FLAG_REDUCTION | (((hist_reduction2 >> 16u) & 0xFFFFu) << 2u))
//         );
//       }
//     }

//     var shift_val: u32 = 1u;
//     for (var jv: u32 = RADIX >> 2u; jv > 0; jv = jv >> 1u) {
//       workgroupBarrier();
//       if (l < jv) {
//         let a_idx = (((((l << 1u) + 2u) << shift_val) - 1u) >> 1u);
//         let b_idx = (((((l << 1u) + 1u) << shift_val) - 1u) >> 1u);
//         atomicAdd(&pass_hist[a_idx], atomicLoad(&pass_hist[b_idx]) & 0xFFFF0000u);
//       }
//       shift_val = shift_val + 1u;
//     }
//     workgroupBarrier();

//     if (l == 0u) {
//       atomicAnd(&pass_hist[HALF_RADIX - 1u], 0xFFFFu);
//     }

//     for (var jv: u32 = 1u; jv < (RADIX >> 1u); jv = jv << 1u) {
//       shift_val = shift_val - 1u;
//       workgroupBarrier();
//       if (l < jv) {
//         let t = (((((l << 1u) + 1u) << shift_val) - 1u) >> 1u);
//         let t2 = (((((l << 1u) + 2u) << shift_val) - 1u) >> 1u);
//         let t3 = atomicLoad(&pass_hist[t]);
//         let t4 = atomicLoad(&pass_hist[t2]);
//         atomicStore(&pass_hist[t], (t3 & 0xFFFFu) | (t4 & 0xFFFF0000u));
//         atomicAdd(&pass_hist[t2], (t3 & 0xFFFF0000u));
//       }
//     }

//     workgroupBarrier();

//     if (l < HALF_RADIX) {
//       let tv = atomicLoad(&pass_hist[l]);
//       atomicStore(&pass_hist[l], (tv >> 16u) + (tv << 16u) + (tv & 0xFFFF0000u));
//     }
//     workgroupBarrier();

//     // Offsets accumulation for packed lanes
//     if (l >= warp_ctx.warp_size * serial_iters) {
//       let tbase = (warp_ctx.warp_id / serial_iters) * HALF_RADIX;
//       for (var ii2: u32 = 0u; ii2 < KEYS_PER_THREAD; ii2 = ii2 + 1u) {
//         let d2 = extract_packed_index(keys[ii2], params.radix_shift);
//         let packed = atomicLoad(&pass_hist[d2 + tbase]) + atomicLoad(&pass_hist[d2]);
//         offs[ii2] = offs[ii2] + extract_packed_value(packed, keys[ii2], params.radix_shift);
//       }
//     } else {
//       for (var ii3: u32 = 0u; ii3 < KEYS_PER_THREAD; ii3 = ii3 + 1u) {
//         let d3 = extract_packed_index(keys[ii3], params.radix_shift);
//         offs[ii3] = offs[ii3] + extract_packed_value(atomicLoad(&pass_hist[d3]), keys[ii3], params.radix_shift);
//       }
//     }

//     if (l < RADIX) {
//       let shift_sel = select(0u, 16u, (l & 1u) != 0u);
//       exclusive_hist_reduction = (atomicLoad(&pass_hist[l >> 1u]) >> shift_sel) & 0xFFFFu;
//     }

//     workgroupBarrier();
//   }

  // -------- Decoupled look-back to get global base per digit --------
  if (l < RADIX) {
    var lookback: u32 = 0u;
    for (var k: i32 = i32(tile_idx); k >= 0;) {
      let flag_payload = atomicLoad(&pass_histogram.data[pass_hist_offset(u32(k)) + l]);
      let flag    = flag_payload & FLAG_MASK;
      let payload = flag_payload >> 2u;

      if (flag == FLAG_INCLUSIVE) {
        lookback = lookback + payload;
        if (tile_idx < (params.thread_blocks - 1u)) {
          // publish REDUCTION to next tile
          let pub_reduction = (FLAG_REDUCTION | (lookback << 2u));
          atomicAdd(&pass_histogram.data[pass_hist_offset(tile_idx + 1u) + l], pub_reduction);
        }
        // store global base (tile prefix - exclusive local)
        digit_base[l] = lookback - exclusive_hist_reduction;
        break;
      }
      
      if (flag == FLAG_REDUCTION) {
        lookback = lookback + payload;
      }
      k = k - 1;
    }
  }
  workgroupBarrier();

  // -------- Direct write-out: each thread writes its own items --------
  if (tile_idx < (params.thread_blocks - 1u)) {
    for (var j: u32 = 0u; j < KEYS_PER_THREAD; j = j + 1u) {
      let key = keys[j];
      let val = vals[j];
      let d   = extract_digit(key, params.radix_shift);
      let dst = digit_base[d] + offs[j];
      scatter_out.data[dst] = key;
      values_scatter_out.data[dst] = val;
    }
  } else {
    // last (partial) tile: skip padded entries
    for (var j: u32 = 0u; j < KEYS_PER_THREAD; j = j + 1u) {
      let key = keys[j];
      if (key != 0xFFFFFFFFu) {
        let val = vals[j];
        let d   = extract_digit(key, params.radix_shift);
        let dst = digit_base[d] + offs[j];
        scatter_out.data[dst] = key;
        values_scatter_out.data[dst] = val;
      }
    }
  }
}
