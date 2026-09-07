// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                         GPU STORAGE HASH MAP                              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║  A compact sidecar turns any equally-sized storage-buffer array into a    ║
// ║  bounded, concurrent hash map. Payload ownership stays with the caller;   ║
// ║  this file owns hashing, probing, timestamps, and atomic claims.          ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// WebGPU guarantees newly created buffers are zero initialized, which is the
// empty-table representation. Reused buffers must be cleared before repurposing.

const HASHMAP_EMPTY_CHECKSUM: u32 = 0u;
const HASHMAP_INVALID_INDEX: u32 = 0xffffffffu;
const HASHMAP_ENTRY_WORD_COUNT: u32 = 3u;

const HASHMAP_RESULT_MISS: u32 = 0u;
const HASHMAP_RESULT_FOUND: u32 = 1u;
const HASHMAP_RESULT_CLAIMED: u32 = 2u;
const HASHMAP_RESULT_ALREADY_UPDATED: u32 = 4u;

// Read-only view used by lookup passes. Mutation passes bind the same buffer as
// array<atomic<u32>> so individual words can participate in atomic claims.
struct HashMapEntry {
    checksum: u32,
    last_used_frame: u32,
    last_update_frame: u32,
};

struct HashMapKey {
    hash_value: u32,
    checksum: u32,
};

struct HashMapResult {
    index: u32,
    status: u32,
};

// PCG is used for table placement. Keeping it independent from the checksum
// mixer prevents a weakness or collision in one hash from defining identity.
fn hashmap_pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// xxHash32's single-word avalanche is used for the independently stored
// checksum. Zero is reserved for empty slots.
fn hashmap_xxhash32(value: u32) -> u32 {
    const PRIME32_2: u32 = 2246822519u;
    const PRIME32_3: u32 = 3266489917u;
    const PRIME32_4: u32 = 668265263u;
    const PRIME32_5: u32 = 374761393u;

    var result = value + PRIME32_5;
    result = PRIME32_4 * ((result << 17u) | (result >> 15u));
    result = PRIME32_2 * (result ^ (result >> 15u));
    result = PRIME32_3 * (result ^ (result >> 13u));
    return result ^ (result >> 16u);
}

fn hashmap_pcg_combine(value: u32, nested_hash: u32) -> u32 {
    return hashmap_pcg(value + nested_hash);
}

fn hashmap_xxhash32_combine(value: u32, nested_hash: u32) -> u32 {
    return hashmap_xxhash32(value + nested_hash);
}

fn hashmap_sanitize_checksum(checksum: u32) -> u32 {
    return select(checksum, 1u, checksum == HASHMAP_EMPTY_CHECKSUM);
}

fn hashmap_search_count(requested_count: u32, capacity: u32) -> u32 {
    return min(max(requested_count, 1u), capacity);
}

fn hashmap_entry_is_expired(
    last_used_frame: u32,
    current_frame: u32,
    lifetime: u32
) -> bool {
    return current_frame - last_used_frame > lifetime;
}

fn hashmap_atomic_word_offset(entry_index: u32, word: u32) -> u32 {
    return entry_index * HASHMAP_ENTRY_WORD_COUNT + word;
}

fn hashmap_atomic_checksum(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32
) -> u32 {
    return atomicLoad(&(*entries)[hashmap_atomic_word_offset(entry_index, 0u)]);
}

fn hashmap_atomic_last_used_frame(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32
) -> u32 {
    return atomicLoad(&(*entries)[hashmap_atomic_word_offset(entry_index, 1u)]);
}

// Claim an empty or expired slot by replacing the checksum directly. Weak
// compare-exchange may fail spuriously, so retry without spinning indefinitely.
fn hashmap_try_claim(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32,
    observed_checksum: u32,
    checksum: u32,
    current_frame: u32,
    lifetime: u32
) -> bool {
    if (
        observed_checksum != HASHMAP_EMPTY_CHECKSUM &&
        !hashmap_entry_is_expired(
            hashmap_atomic_last_used_frame(entries, entry_index),
            current_frame,
            lifetime
        )
    ) {
        return false;
    }

    let checksum_offset = hashmap_atomic_word_offset(entry_index, 0u);
    for (var attempt = 0u; attempt < 4u; attempt = attempt + 1u) {
        let claim = atomicCompareExchangeWeak(
            &(*entries)[checksum_offset],
            observed_checksum,
            hashmap_sanitize_checksum(checksum)
        );
        if (claim.exchanged) {
            atomicStore(
                &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
                current_frame
            );
            atomicStore(
                &(*entries)[hashmap_atomic_word_offset(entry_index, 2u)],
                current_frame
            );
            return true;
        }
        if (claim.old_value != observed_checksum) {
            return false;
        }
    }
    return false;
}

// Checksum-only lookup mirrors the article's compact representation. Clients
// that retain an exact key in their payload can additionally validate it after
// this returns, as SCGI does, making checksum collisions harmless.
fn hashmap_find(
    entries: ptr<storage, array<HashMapEntry>, read>,
    key: HashMapKey,
    capacity: u32,
    requested_search_count: u32
) -> u32 {
    let search_count = hashmap_search_count(requested_search_count, capacity);
    var entry_index = key.hash_value % capacity;
    for (var probe = 0u; probe < search_count; probe = probe + 1u) {
        let checksum = (*entries)[entry_index].checksum;
        if (checksum == key.checksum) {
            return entry_index;
        }
        if (checksum == HASHMAP_EMPTY_CHECKSUM) {
            break;
        }
        entry_index = select(entry_index + 1u, 0u, entry_index + 1u == capacity);
    }
    return HASHMAP_INVALID_INDEX;
}

// Generic find-or-claim for payloads whose 32-bit checksum is their identity.
fn hashmap_find_or_claim(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    key: HashMapKey,
    capacity: u32,
    requested_search_count: u32,
    current_frame: u32,
    lifetime: u32
) -> HashMapResult {
    let search_count = hashmap_search_count(requested_search_count, capacity);
    var reclaim_index = HASHMAP_INVALID_INDEX;
    var reclaim_checksum = HASHMAP_EMPTY_CHECKSUM;
    var entry_index = key.hash_value % capacity;

    for (var probe = 0u; probe < search_count; probe = probe + 1u) {
        let checksum = hashmap_atomic_checksum(entries, entry_index);
        if (checksum == key.checksum) {
            let update_offset = hashmap_atomic_word_offset(entry_index, 2u);
            // Same-frame duplicates dominate feedback traffic. Keep their hot
            // path read-only instead of issuing two contended atomic writes.
            if (atomicLoad(&(*entries)[update_offset]) == current_frame) {
                return HashMapResult(
                    entry_index,
                    HASHMAP_RESULT_ALREADY_UPDATED
                );
            }
            atomicStore(
                &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
                current_frame
            );
            let previous_update_frame = atomicExchange(
                &(*entries)[update_offset],
                current_frame
            );
            if (previous_update_frame == current_frame) {
                return HashMapResult(
                    entry_index,
                    HASHMAP_RESULT_ALREADY_UPDATED
                );
            }
            return HashMapResult(entry_index, HASHMAP_RESULT_FOUND);
        }

        let is_empty = checksum == HASHMAP_EMPTY_CHECKSUM;
        let is_expired = !is_empty && hashmap_entry_is_expired(
                hashmap_atomic_last_used_frame(entries, entry_index),
                current_frame,
                lifetime
            );
        if ((is_empty || is_expired) && reclaim_index == HASHMAP_INVALID_INDEX) {
            reclaim_index = entry_index;
            reclaim_checksum = checksum;
        }
        if (is_empty) {
            break;
        }
        entry_index = select(entry_index + 1u, 0u, entry_index + 1u == capacity);
    }

    if (reclaim_index != HASHMAP_INVALID_INDEX) {
        if (hashmap_try_claim(
            entries,
            reclaim_index,
            reclaim_checksum,
            key.checksum,
            current_frame,
            lifetime
        )) {
            return HashMapResult(reclaim_index, HASHMAP_RESULT_CLAIMED);
        }
        return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_MISS);
    }

    return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_MISS);
}
