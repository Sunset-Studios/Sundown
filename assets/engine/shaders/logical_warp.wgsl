const LOGICAL_WORKGROUP_SIZE   : u32 = 256u;
const LOGICAL_WARP_SIZE        : u32 = 32u;
const NUM_WARPS_256            : u32 = 256u / LOGICAL_WARP_SIZE;

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

fn warp_shuffle_index_u32(c: WarpCtx, value: u32, lane: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();
  return warp_tmp_u32[c.warp_base + lane];
}

fn warp_shuffle_index_f32(c: WarpCtx, value: f32, lane: u32) -> f32 {
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

fn warp_any(c: WarpCtx, predicate: bool) -> bool {
  return any(warp_ballot_u32(c, predicate) != 0u);
}

fn warp_all(c: WarpCtx, predicate: bool) -> bool {
  let mask0 = select(0u, 0xFFFFFFFFu, c.warp_size == 32u);
  let mask1 = select(0u, 0xFFFFFFFFu, c.warp_size == 64u);
  let mask2 = select(0u, 0xFFFFFFFFu, c.warp_size == 128u);
  let mask3 = select(0u, 0xFFFFFFFFu, c.warp_size == 256u);
  return warp_ballot_u32(c, predicate) == vec4<u32>(mask0, mask1, mask2, mask3);
}

// -------- Reductions (sum) --------
fn warp_reduce_add_u32(c: WarpCtx, value: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();

  var step = c.warp_size / 2u;
  loop {
    if (c.lane_id < step) {
      warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + warp_tmp_u32[c.thread_id + step];
    }
    workgroupBarrier();
    if (step == 1u) { break; }
    step = step / 2u;
  }
  // Broadcast final sum from lane 0 (stored at warp_base).
  return warp_tmp_u32[c.warp_base];
}

fn warp_reduce_add_f32(c: WarpCtx, value: f32) -> f32 {
  warp_tmp_f32[c.thread_id] = value;
  workgroupBarrier();

  var step = c.warp_size / 2u;
  loop {
    if (c.lane_id < step) {
      warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + warp_tmp_f32[c.thread_id + step];
    }
    workgroupBarrier();
    if (step == 1u) { break; }
    step = step / 2u;
  }
  return warp_tmp_f32[c.warp_base];
}

// -------- Prefix scans (inclusive/exclusive, +) --------
fn warp_scan_inclusive_add_u32(c: WarpCtx, value: u32) -> u32 {
  warp_tmp_u32[c.thread_id] = value;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= c.warp_size) { break; }
    var addend: u32 = 0u;
    if (c.lane_id >= offset) {
      addend = warp_tmp_u32[c.thread_id - offset];
    }
    workgroupBarrier();
    warp_tmp_u32[c.thread_id] = warp_tmp_u32[c.thread_id] + addend;
    workgroupBarrier();
    offset = offset * 2u;
  }
  return warp_tmp_u32[c.thread_id];
}

fn warp_scan_exclusive_add_u32(c: WarpCtx, value: u32) -> u32 {
  let inc = warp_scan_inclusive_add_u32(c, value);
  return inc - value;
}

fn warp_scan_inclusive_add_f32(c: WarpCtx, value: f32) -> f32 {
  warp_tmp_f32[c.thread_id] = value;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= c.warp_size) { break; }
    var addend: f32 = 0.0;
    if (c.lane_id >= offset) {
      addend = warp_tmp_f32[c.thread_id - offset];
    }
    workgroupBarrier();
    warp_tmp_f32[c.thread_id] = warp_tmp_f32[c.thread_id] + addend;
    workgroupBarrier();
    offset = offset * 2u;
  }
  return warp_tmp_f32[c.thread_id];
}

fn warp_scan_exclusive_add_f32(c: WarpCtx, value: f32) -> f32 {
  let inc = warp_scan_inclusive_add_f32(c, value);
  return inc - value;
}
