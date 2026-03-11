enable subgroups;

const LOGICAL_WORKGROUP_SIZE   : u32 = 256u;
const LOGICAL_WARP_SIZE        : u32 = 32u;
const NUM_WARPS_256            : u32 = 8u; // LOGICAL_WORKGROUP_SIZE / LOGICAL_WARP_SIZE

struct WarpCtx {
  thread_id : u32,   // local_invocation_id.x (still handy)
  lane_id   : u32,   // subgroup_invocation_id
  warp_id   : u32,   // undefined notion natively; keep 0 for symmetry
  warp_size : u32,   // subgroup_size
  warp_base : u32,   // thread_id - lane_id (for compatibility)
}

fn make_warp_ctx(local_tid: u32, lane: u32, warp_size: u32) -> WarpCtx {
  let wid  = local_tid / warp_size;
  return WarpCtx(local_tid, lane, wid, warp_size, wid * warp_size);
}

#define lane_id(local_id, warp_size) (local_id & (warp_size - 1u))
#define warp_id(local_id, warp_size) (local_id / warp_size)

#define is_warp_leader(warp_ctx) (warp_ctx.lane_id == 0u)

#define warp_broadcast_u32(warp_ctx, value, lane) subgroupBroadcast(value, lane)
#define warp_broadcast_f32(warp_ctx, value, lane) subgroupBroadcast(value, lane)

#define warp_broadcast_first_u32(warp_ctx, value) subgroupBroadcastFirst(value)
#define warp_broadcast_first_f32(warp_ctx, value) subgroupBroadcastFirst(value)

#define warp_shuffle_u32(warp_ctx, value, lane) subgroupShuffle(value, lane)
#define warp_shuffle_f32(warp_ctx, value, lane) subgroupShuffle(value, lane)

#define warp_ballot_u32(warp_ctx, predicate) subgroupBallot(predicate)

#define warp_reduce_add_u32(warp_ctx, value) subgroupAdd(value)
#define warp_reduce_add_f32(warp_ctx, value) subgroupAdd(value)

#define warp_scan_inclusive_add_u32(warp_ctx, value) subgroupInclusiveAdd(value)
#define warp_scan_exclusive_add_u32(warp_ctx, value) subgroupExclusiveAdd(value)
#define warp_scan_inclusive_add_f32(warp_ctx, value) subgroupInclusiveAdd(value)
#define warp_scan_exclusive_add_f32(warp_ctx, value) subgroupExclusiveAdd(value)

#define warp_min_u32(warp_ctx, value) subgroupMin(value)
#define warp_max_u32(warp_ctx, value) subgroupMax(value)
#define warp_min_f32(warp_ctx, value) subgroupMin(value)
#define warp_max_f32(warp_ctx, value) subgroupMax(value)

#define warp_any(warp_ctx, predicate) subgroupAny(predicate)
#define warp_all(warp_ctx, predicate) subgroupAll(predicate)

#define warp_or(warp_ctx, value) subgroupOr(value)
#define warp_and(warp_ctx, value) subgroupAnd(value)
#define warp_xor(warp_ctx, value) subgroupXor(value)
