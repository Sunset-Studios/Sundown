#include "common.wgsl"
#include "acceleration_common.wgsl"

////////////////////////////////////////////////////////////////////////////////
// OneSweep (WGSL, logical-warp version when there's no subgroups support)
// Modified and Ported to WGSL (logical warps, no subgroups)
//
// Original: HLSL by Thomas Smith (2024-03-14)
// Based on research by Adinets & Merrill (NVIDIA) - OneSweep (2022)
//
// High-level overview:
// - Stable radix binning/scatter using 8-bit digits (4 passes for 32-bit keys).
// - Avoids global sync by using a decoupled look-back buffer that publishes
//   per-tile reductions and inclusive prefixes, allowing later tiles to derive
//   their global base without a device-wide barrier.
// - Per pass, the algorithm builds per-tile histograms, computes tile-exclusive
//   bases, performs a look-back to add prior tiles, and scatters keys/values.
//
// Memory overview:
// - pass_histogram (device): per-tile/per-digit entries storing either an
//   inclusive prefix (FLAG_INCLUSIVE) or a reduction payload (FLAG_REDUCTION).
// - global_historgram (device): per-pass histogram (all tiles) used to seed
//   the scan kernel (onesweep_scan) which writes inclusive prefixes to
//   pass_histogram in a circular pattern.
// - pass_hist (workgroup): per-tile scratch for counters, staging, and per-digit
//   tile bases used during scatter.
//
// Note: Requires minimum workgroup storage size of 32KB.
////////////////////////////////////////////////////////////////////////////////

// ==================================
// Tunables / Constants
// ==================================
// Controls how work is partitioned across the GPU and how many keys each
// workgroup processes per tile (TILE_SIZE). Adjust PASS_DIM and G_HIST_DIM
// for hardware occupancy tradeoffs; TILE_SIZE must be KEYS_PER_THREAD * PASS_DIM.
const PASS_DIM            : u32 = 256u;    // threads in DigitBinningPass workgroup
const G_HIST_DIM          : u32 = 128u;    // threads in GlobalHistogram workgroup
const TILE_SIZE           : u32 = 4096u;   // size of a tile (KEYS_PER_THREAD * PASS_DIM)

// 8-bit radix parameters (RADIX_LOG = 8) → 4 passes for 32-bit keys.
const RADIX               : u32 = 256u;
const RADIX_MASK          : u32 = 255u;
const RADIX_LOG           : u32 = 8u;
const RADIX_PASSES        : u32 = 4u;
const HALF_RADIX          : u32 = 128u;

const SEC_RADIX_START     : u32 = 256u;
const THIRD_RADIX_START   : u32 = 512u;
const FOURTH_RADIX_START  : u32 = 768u;

const KEYS_PER_THREAD     : u32 = 16u;

// Decoupled look-back flags used in pass_histogram (2 LSBs are flags).
// Upper 30 bits carry payload (inclusive prefix or reduction amount).
const FLAG_NOT_READY      : u32 = 0u;
const FLAG_REDUCTION      : u32 = 1u;
const FLAG_INCLUSIVE      : u32 = 2u;
const FLAG_MASK           : u32 = 3u;

// ==================================
// Data Structures 
// ==================================
// Parameters are shared across kernels for the current pass.
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
// Resources for keys/values, intermediate histogram/flags and control.
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
// Per-tile scratch for per-digit counts and staging during scatter.
var<workgroup> pass_hist          : array<u32, TILE_SIZE + RADIX>;
// Local accumulation of the per-digit global histogram (two halves), then reduced.
var<workgroup> global_hist        : array<atomic<u32>, RADIX * 2u * 4u>;
// Temporary storage for scan of the global histogram per pass plane.
var<workgroup> scan               : array<u32, RADIX>;

// ==================================
// Helper Functions
// ==================================
// Digit extraction and packed helpers used by the binning/scatter.
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

// Current radix pass index in [0, RADIX_PASSES).
fn current_pass() -> u32 {
  return (params.radix_shift >> 3u); // shift / RADIX_LOG
}

// Base offset into pass_histogram for a given tile index at the current pass.
fn pass_hist_offset(tile_index: u32) -> u32 {
  return ((current_pass() * params.thread_blocks) + tile_index) << RADIX_LOG; 
}

fn wave_hists_size_ge16(_c: WarpCtx) -> u32 {
  return PASS_DIM / _c.warp_size * RADIX;
}

fn wave_hists_size_lt16(_c: WarpCtx) -> u32 {
  return TILE_SIZE;
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
    return shared_offset(_c, local_id, serial_iters) + tile_idx * TILE_SIZE;
}

fn global_hist_offset(pass_idx: u32) -> u32 {
    return pass_idx * RADIX;
}

// ==================================
// Kernels
// ==================================
// InitOneSweep: clears pass_histogram, global_historgram, and per-pass tile indices.
@compute @workgroup_size(256)
fn onesweep_init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;

  // Grid-strided clear of per-pass/per-tile histogram flags and payloads.
  // Clear pass_histogram: e_threadBlocks * RADIX * RADIX_PASSES
  let clear_end = params.thread_blocks * RADIX * RADIX_PASSES;
  for (var i: u32 = id; i < clear_end; i += 65536u) {
    atomicStore(&pass_histogram.data[i], 0u);
  }
  // Clear device-wide per-pass global histogram planes.
  // Clear global_historgram: RADIX * RADIX_PASSES
  if (id < RADIX * RADIX_PASSES) {
    atomicStore(&global_historgram.data[id], 0u);
  }
  // Reset per-pass tile indices so each workgroup can atomically claim tiles.
  if (id < RADIX_PASSES) {
    atomicStore(&tile_indices.data[id], 0u);
  }
}

// GlobalHistogram: Accumulate counts for each 8-bit digit across tiles.
// We write into two local halves to reduce contention, then merge and add to
// the device-wide per-pass histogram planes.
@compute @workgroup_size(G_HIST_DIM)
fn onesweep_global_histogram(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>
) {
  let tile_ix = wid.x;
  // 1) Clear shared accumulation buffer used by this workgroup
  let hist_end = RADIX * 2u * 4u;
  for (var i: u32 = lid.x; i < hist_end; i += G_HIST_DIM) {
    atomicStore(&global_hist[i], 0u);
  }
  workgroupBarrier();

  // 2) Pick which half we write to (0..63 first half, 64..127 second half)
  let hist_offset = (lid.x / 64u) * RADIX;
  //    Compute the end of the range for this tile (last tile may be partial)
  let tile_end    = select((tile_ix + 1u) * TILE_SIZE, params.key_count, tile_ix == (params.thread_blocks - 1u));

  // 3) For each key assigned to this workgroup, increment the digit buckets
  //    for all four 8-bit digits (0, 8, 16, 24 bit shifts)
  for (var idx: u32 = lid.x + tile_ix * TILE_SIZE; idx < tile_end; idx += G_HIST_DIM) {
    let t = keys_buffer.data[idx];
    let d0 = extract_digit(t, 0u);
    let d1 = extract_digit(t, 8u);
    let d2 = extract_digit(t, 16u);
    let d3 = extract_digit(t, 24u);
    let b0 = (d0 + hist_offset) * 4u;
    let b1 = (d1 + hist_offset) * 4u;
    let b2 = (d2 + hist_offset) * 4u;
    let b3 = (d3 + hist_offset) * 4u;
    atomicAdd(&global_hist[b0 + 0u], 1u);
    atomicAdd(&global_hist[b1 + 1u], 1u);
    atomicAdd(&global_hist[b2 + 2u], 1u);
    atomicAdd(&global_hist[b3 + 3u], 1u);
  }
  workgroupBarrier();

  // 4) Merge the two halves and add to the device-wide pass planes
  for (var i: u32 = lid.x; i < RADIX; i += G_HIST_DIM) {
    let base0 = i * 4u;
    let base1 = (i + RADIX) * 4u;
    let x_sum = atomicLoad(&global_hist[base0 + 0u]) + atomicLoad(&global_hist[base1 + 0u]);
    let y_sum = atomicLoad(&global_hist[base0 + 1u]) + atomicLoad(&global_hist[base1 + 1u]);
    let z_sum = atomicLoad(&global_hist[base0 + 2u]) + atomicLoad(&global_hist[base1 + 2u]);
    let w_sum = atomicLoad(&global_hist[base0 + 3u]) + atomicLoad(&global_hist[base1 + 3u]);
    atomicAdd(&global_historgram.data[i],                      x_sum);
    atomicAdd(&global_historgram.data[i + SEC_RADIX_START],    y_sum);
    atomicAdd(&global_historgram.data[i + THIRD_RADIX_START],  z_sum);
    atomicAdd(&global_historgram.data[i + FOURTH_RADIX_START], w_sum);
  }
}

// Scan: block-wide exclusive scan of per-digit global histograms (one pass plane).
// Results are written back as inclusive prefixes using a circular index pattern
// with FLAG_INCLUSIVE to seed decoupled look-back in the binning stage.
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

  // 1) Load per-digit counts for this pass plane and compute warp-exclusive scan
  scan[l] = atomicLoad(&global_historgram.data[l + wid.x * RADIX]);
  scan[l] += warp_scan_exclusive_add_u32(warp_ctx, scan[l]);
    
  // 2) Accumulate across warp boundaries to get block-wide exclusive scan
  workgroupBarrier();
  let next_lane = (l + 1u) * warp_ctx.warp_size - 1u;
  let prefix_sum = warp_scan_exclusive_add_u32(warp_ctx, scan[next_lane]);
  if (l < (RADIX / warp_ctx.warp_size)) {
    scan[next_lane] += prefix_sum;
  }
  workgroupBarrier();

  // 3) Write inclusive prefixes into pass_histogram using circular indexing
  //    to seed decoupled look-back in the binning kernel
  let lane_mask = warp_ctx.warp_size - 1u;
  let index = ((warp_ctx.lane_id + 1u) & lane_mask) + (l & ~lane_mask);
  let inclusive_scan = ((select(0u, scan[l], warp_ctx.lane_id != lane_mask) +
        select(0u, warp_broadcast_u32(warp_ctx, scan[l - 1u], 0u), l >= warp_ctx.warp_size)
    ) << 2u) | FLAG_INCLUSIVE;
  atomicStore(&pass_histogram.data[index + wid.x * RADIX * params.thread_blocks], inclusive_scan);
}

// DigitBinningPass: chained scan with decoupled look-back using logical warps (32)
// Steps per tile:
//  (1) Claim tile index, clear tile-local histogram region.
//  (2) Load keys/values for this tile (pad last tile if partial).
//  (3) Build per-digit offsets per lane using ballots (branchless).
//  (4) Reduce to tile-exclusive digit prefixes across the workgroup.
//  (5) Publish reductions and inclusive prefixes for look-back consumption.
//  (6) Perform look-back to compute the global base for each digit.
//  (7) Scatter keys/values directly to the global destinations (stable order).
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
 
  // We assume warp sizes that are generally 16 or larger.
  // Very rare to get warp sizes that are smaller than that.

  // (1) Clear tile-local wave histogram region used for reductions
  let hist_end = wave_hists_size_ge16(warp_ctx);
  for (var i: u32 = l; i < hist_end; i += PASS_DIM) {
    pass_hist[i] = 0u;
  }
  // Claim a unique tile index for this workgroup in the current pass
  if (l == 0u) {
    pass_hist[TILE_SIZE - 1u] = atomicAdd(&tile_indices.data[current_pass()], 1u);
  }
  workgroupBarrier();

  let tile_idx = pass_hist[TILE_SIZE - 1u];
  
  // (2) Load keys/values for this tile; offs[] holds tile-local scatter indices
  var keys: array<u32, KEYS_PER_THREAD>;
  var vals: array<u32, KEYS_PER_THREAD>;
  var offs: array<u32, KEYS_PER_THREAD>;

  let serial_iters: u32 = (PASS_DIM / warp_ctx.warp_size + 31u) / 32u;
  let base = device_offset(warp_ctx, l, tile_idx, serial_iters);

  if (tile_idx < (params.thread_blocks - 1u)) {
    var t: u32 = base;
    for (var j0: u32 = 0u; j0 < KEYS_PER_THREAD; j0 = j0 + 1) {
      keys[j0] = keys_buffer.data[t];
      vals[j0] = values_buffer.data[t];
      t += warp_ctx.warp_size * serial_iters;
    }
  }
  
  if (tile_idx == (params.thread_blocks - 1u)) {
    // Last tile may be partial; pad with sentinel values outside range
    var t2: u32 = base;
    for (var j1: u32 = 0u; j1 < KEYS_PER_THREAD; j1 = j1 + 1u) {
      let in_range = t2 < params.key_count;
      keys[j1] = select(0xffffffffu, keys_buffer.data[t2], in_range);
      vals[j1] = select(0xffffffffu, values_buffer.data[t2], in_range);
      t2 += warp_ctx.warp_size * serial_iters;
    }
  }

  // (3) For each thread's KEYS_PER_THREAD, compute per-digit lane-local offsets
  //     using ballots to count preceding lanes with the same digit
  var exclusive_hist_reduction: u32 = 0u;
  let wave_parts = (warp_ctx.warp_size + 31u) / 32u;
  for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
    // Start with all bits set for each 32-lane part
    var wave_flags = vec4<u32>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);

    // Intersect ballots across the 8 bits of the digit to isolate lanes
    // whose digit equals this lane's digit
    for (var j: u32 = 0u; j < RADIX_LOG; j = j + 1u) {
      let t = ((keys[i] >> (j + params.radix_shift)) & 1u) != 0u;
      let ballot = warp_ballot_u32(warp_ctx, t);
      for (var k: u32 = 0u; k < wave_parts; k = k + 1u) {
        let mask = (select(0xFFFFFFFFu, 0u, t) ^ ballot[k]);
        wave_flags[k] = wave_flags[k] & mask;
      }
    }

    // Count matching lanes strictly before this lane (branchless across parts)
    var bits: u32 = 0u;
    for (var k: u32 = 0u; k < wave_parts; k = k + 1u) {
      if (warp_ctx.lane_id >= (k * 32u)) {
        let lt_mask = select(
          (1u << (warp_ctx.lane_id & 31u)) - 1u,
          0xFFFFFFFFu,
          warp_ctx.lane_id >= ((k + 1u) * 32u)
        );
        bits = bits + countOneBits(wave_flags[k] & lt_mask);
      }
    }

    let index = extract_digit(keys[i], params.radix_shift) + (warp_ctx.warp_id * RADIX);
    offs[i] = pass_hist[index] + bits;

    workgroupBarrier();

    if (bits == 0u) {
      for (var k: u32 = 0u; k < wave_parts; k = k + 1u) {
        pass_hist[index] = pass_hist[index] + countOneBits(wave_flags[k]);
      }
    }
    workgroupBarrier();
  }

  // (4) Reduce per-digit counts across the wave-hists region to get tile-exclusive prefixes
  var hist_reduction: u32 = 0u;
  if (l < RADIX) {
    hist_reduction = pass_hist[l];
    let wave_hist_size = wave_hists_size_ge16(warp_ctx);
    for (var i: u32 = l + RADIX; i < wave_hist_size; i = i + RADIX) {
      let hist_i = pass_hist[i];
      hist_reduction += hist_i;
      pass_hist[i] = hist_reduction - hist_i;
    }

    if (tile_idx < (params.thread_blocks - 1u)) {
      // Publish reduction payload (no flag bit overlap) to look-back buffer of next tile
      let value = (hist_reduction << 2u) | FLAG_REDUCTION;
      atomicAdd(&pass_histogram.data[l + pass_hist_offset(tile_idx + 1u)], value);
    }
  }
  hist_reduction = select(hist_reduction, hist_reduction + warp_scan_exclusive_add_u32(warp_ctx, hist_reduction), l < RADIX);

  workgroupBarrier();

  // (5) Circularly scatter per-digit inclusive prefixes to pass_hist
  if (l < RADIX) {
    let lane_mask = warp_ctx.warp_size - 1u;
    let dst = ((warp_ctx.lane_id + 1u) & lane_mask) + (l & ~lane_mask);
    pass_hist[dst] = hist_reduction;
  }
  workgroupBarrier();

  // Stitch warp boundaries by scanning warp leaders and broadcasting
  let idx = l * warp_ctx.warp_size;
  let boundary_val = pass_hist[idx];
  let boundary_excl = warp_scan_exclusive_add_u32(warp_ctx, boundary_val);
  if (l < (RADIX / warp_ctx.warp_size)) {
    pass_hist[idx] = boundary_excl;
  }
  workgroupBarrier();

  let prev_from_lane1 = warp_shuffle_u32(warp_ctx, pass_hist[l - 1u], 1u);
  if (l < RADIX && li != 0u) {
    pass_hist[l] += prev_from_lane1;
  }
  workgroupBarrier();

  // (6) Add per-lane base to each thread's pending offsets
  if (l >= warp_ctx.warp_size) {
    let t = warp_ctx.warp_id * RADIX;
    for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
      let d = extract_digit(keys[i], params.radix_shift);
      offs[i] += pass_hist[d + t] + pass_hist[d];
    }
  } else {
    for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
      let d = extract_digit(keys[i], params.radix_shift);
      offs[i] += pass_hist[d];
    }
  }

  if (l < RADIX) {
    exclusive_hist_reduction = pass_hist[l];
  }
  workgroupBarrier();

  // Stage keys at their tile-local destination so each thread can scatter contiguously
  for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
    pass_hist[offs[i]] = keys[i];
  }
  
  // (7) Decoupled look-back: accumulate prior tiles' totals to form the global base
  if (l < RADIX) {
    var lookback: u32 = 0u;
    for (var k: i32 = i32(tile_idx); k >= 0;) {
      let flag_payload = atomicLoad(&pass_histogram.data[pass_hist_offset(u32(k)) + l]);
      let flag    = flag_payload & FLAG_MASK;
      let payload = flag_payload >> 2u;

      if (flag == FLAG_INCLUSIVE) {
        // Inclusive prefix published by onesweep_scan or a prior tile
        lookback += payload;
        if (tile_idx < (params.thread_blocks - 1u)) {
          // Publish total through this tile so later tiles can stop early
          let val = (lookback << 2u) | FLAG_REDUCTION;
          atomicAdd(&pass_histogram.data[pass_hist_offset(tile_idx + 1u) + l], val);
        }
        // Global base for this digit = prefix up to prev tile(s)
        pass_hist[l + TILE_SIZE] = lookback - exclusive_hist_reduction;
        break;
      }
      
      if (flag == FLAG_REDUCTION) {
        lookback += payload;
        k--;
      }
    }
  }
  workgroupBarrier();

  // (8) Direct scatter using per-thread offsets and tile-digit bases
  if (tile_idx < (params.thread_blocks - 1u)) {
    for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
      let d = extract_digit(keys[i], params.radix_shift);
      let base_idx = pass_hist[d + TILE_SIZE] + offs[i];
      scatter_out.data[base_idx] = keys[i];
      values_scatter_out.data[base_idx] = vals[i];
    }
  }

  // Last (partial) tile: guard scatter by source index in range
  if (tile_idx == (params.thread_blocks - 1u)) {
    let stride = warp_ctx.warp_size * serial_iters;
    var t_src: u32 = base;
    for (var i: u32 = 0u; i < KEYS_PER_THREAD; i = i + 1u) {
      if (t_src < params.key_count) {
        let d = extract_digit(keys[i], params.radix_shift);
        let base_idx = pass_hist[d + TILE_SIZE] + offs[i];
        scatter_out.data[base_idx] = keys[i];
        values_scatter_out.data[base_idx] = vals[i];
      }
      t_src = t_src + stride;
    }
  }
}
