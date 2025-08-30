const LOGICAL_WORKGROUP_SIZE   : u32 = 256u;
const LOGICAL_WARP_SIZE        : u32 = 32u;
const NUM_WARPS_256            : u32 = 8u; // LOGICAL_WORKGROUP_SIZE / LOGICAL_WARP_SIZE

struct WarpCtx {
  thread_id : u32,   // local_invocation_id.x
  lane_id   : u32,   // thread_id % LOGICAL_WARP_SIZE
  warp_id   : u32,   // thread_id / LOGICAL_WARP_SIZE
  warp_size : u32,   // = LOGICAL_WARP_SIZE
  warp_base : u32,   // warp_id * LOGICAL_WARP_SIZE
};

// Scratch (one slot per thread in the workgroup)
var<workgroup> warp_tmp_u32 : array<u32, LOGICAL_WORKGROUP_SIZE>;
var<workgroup> warp_tmp_f32 : array<f32, LOGICAL_WORKGROUP_SIZE>;
// Workgroup-wide reduction helpers (atomic-based)
var<workgroup> wg_reduce_min_flag  : atomic<u32>;
var<workgroup> wg_reduce_max_flag  : atomic<u32>;
var<workgroup> wg_reduce_min_value : atomic<u32>;
var<workgroup> wg_reduce_max_value : atomic<u32>;

fn make_warp_ctx(local_tid: u32, lane: u32, warp_size: u32) -> WarpCtx {
  let wid  = local_tid / warp_size;
  return WarpCtx(local_tid, lane, wid, warp_size, wid * warp_size);
}

#define lane_id(local_id, warp_size) (local_id & (warp_size - 1u))
#define warp_id(local_id, warp_size) (local_id / warp_size)

#define is_warp_leader(warp_ctx) (warp_ctx.lane_id == 0u)

// -------- Broadcast / Shuffle (index-based) --------
fn warp_broadcast_u32(c: WarpCtx, value: u32, lane: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();
  return warp_tmp_u32[c.warp_base + lane];
}

fn warp_broadcast_f32(c: WarpCtx, value: f32, lane: u32) -> f32 {
  warp_tmp_f32[c.thread_id] = value;
  workgroupBarrier();
  return warp_tmp_f32[c.warp_base + lane];
}

fn warp_broadcast_first_u32(c: WarpCtx, value: u32) -> u32 {
  // Zero per-warp active flags
  if (is_warp_leader(c)) {
    for (var i: u32 = 0u; i < c.warp_size; i = i + 1u) {
      warp_tmp_f32[c.warp_base + i] = 0.0;
    }
  }
  workgroupBarrier();

  // Mark active lanes and stash values
  warp_tmp_u32[c.thread_id] = value;
  warp_tmp_f32[c.thread_id] = 1.0;
  workgroupBarrier();

  // Elect the lowest active lane
  if (is_warp_leader(c)) {
    var first_lane: u32 = 0u;
    for (var i: u32 = 0u; i < c.warp_size; i = i + 1u) {
      if (warp_tmp_f32[c.warp_base + i] != 0.0) {
        first_lane = i;
        break;
      }
    }
    // Share elected lane id via u32 scratch header
    warp_tmp_u32[c.warp_base] = first_lane;
  }
  workgroupBarrier();

  let src_lane = warp_tmp_u32[c.warp_base];
  return warp_tmp_u32[c.warp_base + src_lane];
}

fn warp_broadcast_first_f32(c: WarpCtx, value: f32) -> f32 {
  // Zero per-warp active flags
  if (is_warp_leader(c)) {
    for (var i: u32 = 0u; i < c.warp_size; i = i + 1u) {
      warp_tmp_u32[c.warp_base + i] = 0u;
    }
  }
  workgroupBarrier();

  // Mark active lanes and stash values
  warp_tmp_f32[c.thread_id] = value;
  warp_tmp_u32[c.thread_id] = 1u;
  workgroupBarrier();

  // Elect the lowest active lane
  if (is_warp_leader(c)) {
    var first_lane: u32 = 0u;
    for (var i: u32 = 0u; i < c.warp_size; i = i + 1u) {
      if (warp_tmp_u32[c.warp_base + i] != 0u) {
        first_lane = i;
        break;
      }
    }
    // Share elected lane id via u32 scratch header
    warp_tmp_u32[c.warp_base] = first_lane;
  }
  workgroupBarrier();

  let src_lane = warp_tmp_u32[c.warp_base];
  return warp_tmp_f32[c.warp_base + src_lane];
}

fn warp_shuffle_u32(c: WarpCtx, value: u32, lane: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();
  return warp_tmp_u32[c.warp_base + lane];
}

fn warp_shuffle_f32(c: WarpCtx, value: f32, lane: u32) -> f32 {
  warp_tmp_f32[c.thread_id] = value;
  workgroupBarrier();
  return warp_tmp_f32[c.warp_base + lane];
}

// -------- Ballot / Any / All (<=32 lanes) --------
fn warp_ballot_u32(c: WarpCtx, predicate: bool) -> vec4<u32> {
  // Each lane writes its bit (within its 32-lane sub-part)
  let sub_lane = c.lane_id & 31u;
  let bit = select(0u, 1u, predicate) << sub_lane;
  warp_tmp_u32[c.thread_id] = bit;
  workgroupBarrier();

  // Warp leader OR-reduces into 4x 32-bit words (parts 0..3)
  var m0: u32 = 0u;
  var m1: u32 = 0u;
  var m2: u32 = 0u;
  var m3: u32 = 0u;
  if (is_warp_leader(c)) {
    for (var i: u32 = 0u; i < c.warp_size; i = i + 1u) {
      let v = warp_tmp_u32[c.warp_base + i];
      let part = i >> 5u;
      if (part == 0u) {
        m0 = m0 | v;
      } else if (part == 1u) {
        m1 = m1 | v;
      } else if (part == 2u) {
        m2 = m2 | v;
      } else {
        m3 = m3 | v;
      }
    }
    // Stash so all lanes can read
    warp_tmp_u32[c.warp_base + 0u] = m0;
    warp_tmp_u32[c.warp_base + 1u] = m1;
    warp_tmp_u32[c.warp_base + 2u] = m2;
    warp_tmp_u32[c.warp_base + 3u] = m3;
  }
  workgroupBarrier();

  return vec4<u32>(
    warp_tmp_u32[c.warp_base + 0u],
    warp_tmp_u32[c.warp_base + 1u],
    warp_tmp_u32[c.warp_base + 2u],
    warp_tmp_u32[c.warp_base + 3u]
  );
}

fn warp_any(warp_ctx: WarpCtx, predicate: bool) -> bool {
  return any(warp_ballot_u32(warp_ctx, predicate) != vec4<u32>(0u, 0u, 0u, 0u));
}

fn warp_all(warp_ctx: WarpCtx, predicate: bool) -> bool {
  let mask0 = select(0u, 0xFFFFFFFFu, warp_ctx.warp_size == 32u);
  let mask1 = select(0u, 0xFFFFFFFFu, warp_ctx.warp_size == 64u);
  let mask2 = select(0u, 0xFFFFFFFFu, warp_ctx.warp_size == 128u);
  let mask3 = select(0u, 0xFFFFFFFFu, warp_ctx.warp_size == 256u);
  return all(warp_ballot_u32(warp_ctx, predicate) == vec4<u32>(mask0, mask1, mask2, mask3));
}

// -------- Reductions (sum) --------
fn warp_reduce_add_u32(c: WarpCtx, value: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();
  // Unrolled for LOGICAL_WARP_SIZE == 32
  if (c.lane_id < 16u) { warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + warp_tmp_u32[c.thread_id + 16u]; }
  workgroupBarrier();
  if (c.lane_id < 8u)  { warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + warp_tmp_u32[c.thread_id + 8u]; }
  workgroupBarrier();
  if (c.lane_id < 4u)  { warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + warp_tmp_u32[c.thread_id + 4u]; }
  workgroupBarrier();
  if (c.lane_id < 2u)  { warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + warp_tmp_u32[c.thread_id + 2u]; }
  workgroupBarrier();
  if (c.lane_id < 1u)  { warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + warp_tmp_u32[c.thread_id + 1u]; }
  workgroupBarrier();
  // Broadcast final sum from lane 0 (stored at warp_base).
  return warp_tmp_u32[c.warp_base];
}

fn warp_reduce_add_f32(c: WarpCtx, value: f32) -> f32 {
  warp_tmp_f32[c.thread_id] = value;
  workgroupBarrier();
  // Unrolled for LOGICAL_WARP_SIZE == 32
  if (c.lane_id < 16u) { warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + warp_tmp_f32[c.thread_id + 16u]; }
  workgroupBarrier();
  if (c.lane_id < 8u)  { warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + warp_tmp_f32[c.thread_id + 8u]; }
  workgroupBarrier();
  if (c.lane_id < 4u)  { warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + warp_tmp_f32[c.thread_id + 4u]; }
  workgroupBarrier();
  if (c.lane_id < 2u)  { warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + warp_tmp_f32[c.thread_id + 2u]; }
  workgroupBarrier();
  if (c.lane_id < 1u)  { warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + warp_tmp_f32[c.thread_id + 1u]; }
  workgroupBarrier();
  return warp_tmp_f32[c.warp_base];
}

// -------- Prefix scans (inclusive/exclusive, +) --------
fn warp_scan_inclusive_add_u32(c: WarpCtx, value: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();
  // Unrolled offsets for LOGICAL_WARP_SIZE == 32
  var addend: u32;
  addend = select(0u, warp_tmp_u32[c.thread_id - 1u],  c.lane_id >= 1u);  workgroupBarrier();
  warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + addend;        workgroupBarrier();
  addend = select(0u, warp_tmp_u32[c.thread_id - 2u],  c.lane_id >= 2u);  workgroupBarrier();
  warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + addend;        workgroupBarrier();
  addend = select(0u, warp_tmp_u32[c.thread_id - 4u],  c.lane_id >= 4u);  workgroupBarrier();
  warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + addend;        workgroupBarrier();
  addend = select(0u, warp_tmp_u32[c.thread_id - 8u],  c.lane_id >= 8u);  workgroupBarrier();
  warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + addend;        workgroupBarrier();
  addend = select(0u, warp_tmp_u32[c.thread_id - 16u], c.lane_id >= 16u); workgroupBarrier();
  warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + addend;        workgroupBarrier();
  return warp_tmp_u32[c.thread_id];
}

fn warp_scan_exclusive_add_u32(c: WarpCtx, value: u32) -> u32 {
  let inc = warp_scan_inclusive_add_u32(c, value);
  return inc - value;
}

fn warp_scan_inclusive_add_f32(c: WarpCtx, value: f32) -> f32 {
  warp_tmp_f32[c.thread_id] = value;
  workgroupBarrier();
  // Unrolled offsets for LOGICAL_WARP_SIZE == 32
  var addend: f32;
  addend = select(0.0, warp_tmp_f32[c.thread_id - 1u],  c.lane_id >= 1u);  workgroupBarrier();
  warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + addend;         workgroupBarrier();
  addend = select(0.0, warp_tmp_f32[c.thread_id - 2u],  c.lane_id >= 2u);  workgroupBarrier();
  warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + addend;         workgroupBarrier();
  addend = select(0.0, warp_tmp_f32[c.thread_id - 4u],  c.lane_id >= 4u);  workgroupBarrier();
  warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + addend;         workgroupBarrier();
  addend = select(0.0, warp_tmp_f32[c.thread_id - 8u],  c.lane_id >= 8u);  workgroupBarrier();
  warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + addend;         workgroupBarrier();
  addend = select(0.0, warp_tmp_f32[c.thread_id - 16u], c.lane_id >= 16u); workgroupBarrier();
  warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + addend;         workgroupBarrier();
  return warp_tmp_f32[c.thread_id];
}

fn warp_scan_exclusive_add_f32(c: WarpCtx, value: f32) -> f32 {
  let inc = warp_scan_inclusive_add_f32(c, value);
  return inc - value;
}

fn warp_min_u32(c: WarpCtx, value: u32) -> u32 {
  // Elect a leader to initialize the accumulator once per call
  let leader = atomicCompareExchangeWeak(&wg_reduce_min_flag, 0u, 1u);
  if (leader.exchanged) {
    atomicStore(&wg_reduce_min_value, value);
  }
  // Ensure accumulator is initialized before updates
  workgroupBarrier();
  // Contribute this lane's value
  atomicMin(&wg_reduce_min_value, value);
  workgroupBarrier();
  let result = atomicLoad(&wg_reduce_min_value);
  // Reset for reuse (only the elected leader clears the flag)
  if (leader.exchanged) {
    atomicStore(&wg_reduce_min_flag, 0u);
  }
  workgroupBarrier();
  return result;
}

fn warp_min_f32(c: WarpCtx, value: f32) -> f32 {
  // Elect a leader to initialize the accumulator once per call
  let leader = atomicCompareExchangeWeak(&wg_reduce_min_flag, 0u, 1u);
  if (leader.exchanged) {
    atomicStore(&wg_reduce_min_value, bitcast<u32>(value));
  }
  // Ensure accumulator is initialized before updates
  workgroupBarrier();
  // Contribute this lane's value
  atomicMin(&wg_reduce_min_value, bitcast<u32>(value));
  workgroupBarrier();
  let result = bitcast<f32>(atomicLoad(&wg_reduce_min_value));
  // Reset for reuse (only the elected leader clears the flag)
  if (leader.exchanged) {
    atomicStore(&wg_reduce_min_flag, 0u);
  }
  workgroupBarrier();
  return result;
}

fn warp_max_u32(c: WarpCtx, value: u32) -> u32 {
  // Elect a leader to initialize the accumulator once per call
  let leader = atomicCompareExchangeWeak(&wg_reduce_max_flag, 0u, 1u);
  if (leader.exchanged) {
    atomicStore(&wg_reduce_max_value, value);
  }
  // Ensure accumulator is initialized before updates
  workgroupBarrier();
  // Contribute this lane's value
  atomicMax(&wg_reduce_max_value, value);
  workgroupBarrier();
  let result = atomicLoad(&wg_reduce_max_value);
  // Reset for reuse (only the elected leader clears the flag)
  if (leader.exchanged) {
    atomicStore(&wg_reduce_max_flag, 0u);
  }
  workgroupBarrier();
  return result;
}

fn warp_max_f32(c: WarpCtx, value: f32) -> f32 {
  // Elect a leader to initialize the accumulator once per call
  let leader = atomicCompareExchangeWeak(&wg_reduce_max_flag, 0u, 1u);
  if (leader.exchanged) {
    atomicStore(&wg_reduce_max_value, bitcast<u32>(value));
  }
  // Ensure accumulator is initialized before updates
  workgroupBarrier();
  // Contribute this lane's value
  atomicMax(&wg_reduce_max_value, bitcast<u32>(value));
  workgroupBarrier();
  let result = bitcast<f32>(atomicLoad(&wg_reduce_max_value));
  // Reset for reuse (only the elected leader clears the flag)
  if (leader.exchanged) {
    atomicStore(&wg_reduce_max_flag, 0u);
  }
  workgroupBarrier();
  return result;
}