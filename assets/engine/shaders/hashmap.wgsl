// =============================================================================
// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║                         GPU STORAGE HASH MAP                              ║
// ╠═══════════════════════════════════════════════════════════════════════════╣
// ║  A compact sidecar turns any equally-sized storage-buffer array into a    ║
// ║  bounded, concurrent hash map. Payload ownership stays with the caller;   ║
// ║  this file owns hashing, probing, timestamps, claims, and publication.    ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
// =============================================================================

// WebGPU guarantees newly created buffers are zero initialized, which is the
// empty-table representation. Reused buffers must be cleared before repurposing.

const HASHMAP_EMPTY_CHECKSUM: u32 = 0u;
const HASHMAP_LOCKED_CHECKSUM: u32 = 0xffffffffu;
const HASHMAP_INVALID_INDEX: u32 = 0xffffffffu;
const HASHMAP_ENTRY_WORD_COUNT: u32 = 4u;

const HASHMAP_RESULT_MISS: u32 = 0u;
const HASHMAP_RESULT_FOUND: u32 = 1u;
const HASHMAP_RESULT_CLAIMED: u32 = 2u;
const HASHMAP_RESULT_BUSY: u32 = 3u;
const HASHMAP_RESULT_ALREADY_UPDATED: u32 = 4u;

// Read-only view used by lookup passes. Mutation passes bind the same buffer as
// array<atomic<u32>> so individual words can participate in atomic claims.
struct HashMapEntry {
    checksum: u32,
    last_used_frame: u32,
    last_update_frame: u32,
    lock: u32,
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
// checksum. Zero and all-ones are reserved for empty and publishing slots.
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
    if (checksum == HASHMAP_EMPTY_CHECKSUM) {
        return 1u;
    }
    if (checksum == HASHMAP_LOCKED_CHECKSUM) {
        return HASHMAP_LOCKED_CHECKSUM - 1u;
    }
    return checksum;
}

// Linear probing intentionally keeps adjacent reads cache coherent. Wrapping
// gives every key the full bounded search window, including keys near the end.
fn hashmap_probe_index(hash_value: u32, probe: u32, capacity: u32) -> u32 {
    return ((hash_value % capacity) + probe) % capacity;
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

fn hashmap_atomic_last_update_frame(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32
) -> u32 {
    return atomicLoad(&(*entries)[hashmap_atomic_word_offset(entry_index, 2u)]);
}

fn hashmap_try_acquire_lock(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32
) -> bool {
    let lock_offset = hashmap_atomic_word_offset(entry_index, 3u);
    // Weak compare-exchange may fail spuriously. A small bounded retry avoids
    // an unbounded GPU spin while making a false busy result very unlikely.
    for (var attempt = 0u; attempt < 4u; attempt = attempt + 1u) {
        let claim = atomicCompareExchangeWeak(&(*entries)[lock_offset], 0u, 1u);
        if (claim.exchanged) {
            return true;
        }
        if (claim.old_value != 0u) {
            return false;
        }
    }
    return false;
}

fn hashmap_release_lock(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32
) {
    atomicStore(&(*entries)[hashmap_atomic_word_offset(entry_index, 3u)], 0u);
}

// Begin writing either an empty slot or an expired occupied slot. The checksum
// is changed to the publishing sentinel only after exclusive ownership and a
// second age check, preventing two invocations from recycling the same entry.
fn hashmap_try_begin_claim(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32,
    observed_checksum: u32,
    current_frame: u32,
    lifetime: u32
) -> bool {
    if (
        observed_checksum == HASHMAP_LOCKED_CHECKSUM ||
        !hashmap_try_acquire_lock(entries, entry_index)
    ) {
        return false;
    }

    let current_checksum = hashmap_atomic_checksum(entries, entry_index);
    if (current_checksum != observed_checksum) {
        hashmap_release_lock(entries, entry_index);
        return false;
    }

    if (
        current_checksum != HASHMAP_EMPTY_CHECKSUM &&
        !hashmap_entry_is_expired(
            hashmap_atomic_last_used_frame(entries, entry_index),
            current_frame,
            lifetime
        )
    ) {
        hashmap_release_lock(entries, entry_index);
        return false;
    }

    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 0u)],
        HASHMAP_LOCKED_CHECKSUM
    );
    return true;
}

// The caller initializes its ordinary payload buffer while the claim is held,
// then publishes the checksum and releases the slot with this function.
fn hashmap_publish_claim(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32,
    checksum: u32,
    current_frame: u32
) {
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
        current_frame
    );
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 2u)],
        current_frame
    );
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 0u)],
        hashmap_sanitize_checksum(checksum)
    );
    hashmap_release_lock(entries, entry_index);
}

fn hashmap_touch_locked_entry(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32,
    current_frame: u32
) {
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
        current_frame
    );
}

fn hashmap_mark_updated_locked(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32,
    current_frame: u32
) {
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 2u)],
        current_frame
    );
}

fn hashmap_clear_locked_entry(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    entry_index: u32
) {
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
        0u
    );
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 2u)],
        0u
    );
    atomicStore(
        &(*entries)[hashmap_atomic_word_offset(entry_index, 0u)],
        HASHMAP_EMPTY_CHECKSUM
    );
    hashmap_release_lock(entries, entry_index);
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
    for (var probe = 0u; probe < search_count; probe = probe + 1u) {
        let entry_index = hashmap_probe_index(key.hash_value, probe, capacity);
        let checksum = (*entries)[entry_index].checksum;
        if (checksum == key.checksum) {
            return entry_index;
        }
        if (checksum == HASHMAP_EMPTY_CHECKSUM) {
            break;
        }
    }
    return HASHMAP_INVALID_INDEX;
}

// Generic find-or-claim for payloads whose 32-bit checksum is their identity.
// A CLAIMED result keeps the slot locked until hashmap_publish_claim() runs.
fn hashmap_find_or_claim(
    entries: ptr<storage, array<atomic<u32>>, read_write>,
    key: HashMapKey,
    capacity: u32,
    requested_search_count: u32,
    current_frame: u32,
    lifetime: u32
) -> HashMapResult {
    let search_count = hashmap_search_count(requested_search_count, capacity);
    for (var probe = 0u; probe < search_count; probe = probe + 1u) {
        let entry_index = hashmap_probe_index(key.hash_value, probe, capacity);
        let checksum = hashmap_atomic_checksum(entries, entry_index);
        if (checksum == key.checksum) {
            if (hashmap_atomic_last_update_frame(entries, entry_index) == current_frame) {
                atomicStore(
                    &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
                    current_frame
                );
                return HashMapResult(
                    entry_index,
                    HASHMAP_RESULT_ALREADY_UPDATED
                );
            }
            if (!hashmap_try_acquire_lock(entries, entry_index)) {
                return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_BUSY);
            }
            if (hashmap_atomic_checksum(entries, entry_index) != key.checksum) {
                hashmap_release_lock(entries, entry_index);
                return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_BUSY);
            }
            atomicStore(
                &(*entries)[hashmap_atomic_word_offset(entry_index, 1u)],
                current_frame
            );
            let previous_update_frame = atomicExchange(
                &(*entries)[hashmap_atomic_word_offset(entry_index, 2u)],
                current_frame
            );
            if (previous_update_frame == current_frame) {
                hashmap_release_lock(entries, entry_index);
                return HashMapResult(
                    entry_index,
                    HASHMAP_RESULT_ALREADY_UPDATED
                );
            }
            hashmap_release_lock(entries, entry_index);
            return HashMapResult(entry_index, HASHMAP_RESULT_FOUND);
        }
        if (checksum == HASHMAP_LOCKED_CHECKSUM) {
            return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_BUSY);
        }
        if (
            checksum == HASHMAP_EMPTY_CHECKSUM ||
            hashmap_entry_is_expired(
                hashmap_atomic_last_used_frame(entries, entry_index),
                current_frame,
                lifetime
            )
        ) {
            if (hashmap_try_begin_claim(
                entries,
                entry_index,
                checksum,
                current_frame,
                lifetime
            )) {
                return HashMapResult(entry_index, HASHMAP_RESULT_CLAIMED);
            }
            return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_BUSY);
        }
    }
    return HashMapResult(HASHMAP_INVALID_INDEX, HASHMAP_RESULT_MISS);
}
