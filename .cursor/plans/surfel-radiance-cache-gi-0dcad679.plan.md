<!-- 0dcad679-6122-4253-a746-c3557fa11cca da74d9ee-2e21-4ac5-8fb5-40e0b198b3fc -->
# Surfel-Based Radiance Caching GI System

## Overview

Replace the stub ReSTIR GI system with a production-ready surfel cache that:

- Extracts surfels dynamically from G-buffer each frame
- Maintains persistent surfel pool with spatial hash acceleration structure
- Updates subset of surfels per frame via path tracing (amortized)
- Propagates radiance between neighbors for infinite bounce GI
- Provides efficient spatial queries during deferred lighting

## Core Data Structures

### 1. Surfel Structure (WGSL)

Create `assets/engine/shaders/gi/surfel_common.wgsl`:

```wgsl
struct Surfel {
    position_radius: vec4<f32>,           // xyz=position, w=radius
    normal_confidence: vec4<f32>,         // xyz=normal, w=confidence (0-1)
    radiance_m: vec4<f32>,                // rgb=cached radiance, w=sample count
    direction_timestamp: vec4<f32>,       // xyz=dominant bounce dir, w=last_update_frame
    albedo_roughness: vec4<f32>,          // rgb=albedo, w=roughness
}

struct SurfelParams {
    max_surfels: u32,
    active_surfel_count: u32,
    update_budget_per_frame: u32,
    grid_cell_size: f32,
    grid_dimensions: vec3<u32>,
    propagation_radius: f32,
    radiance_decay: f32,
    confidence_threshold: f32,
}
```

### 2. Spatial Hash Grid

Use fixed-size 3D grid for O(1) neighbor queries:

- Grid resolution configurable (e.g., 64x64x64 cells)
- Each cell stores list of surfel indices
- Cell size based on expected surfel density

## Implementation Steps

### Step 1: Surfel Manager (JavaScript)

Create `engine/src/renderer/global_illumination/surfel_cache.js`:

- Manages surfel buffer (storage buffer with max capacity, e.g., 50k surfels)
- Tracks active surfel count and free list
- Provides configuration interface (max_surfels, update_budget, grid params)
- Integrates with existing `gi.js` class structure

### Step 2: Surfel Spawning Pass

Shader: `assets/engine/shaders/gi/surfel_spawn.wgsl`

- Read G-buffer (position, normal, albedo, roughness)
- Hash each pixel's world position to spatial grid cell
- Check if cell needs new surfel (compare with existing)
- Spawn surfels in undersampled regions
- Initialize with G-buffer material properties
- Use atomic operations for insertion into grid cells

### Step 3: Surfel Culling/Compaction Pass

Shader: `assets/engine/shaders/gi/surfel_cull.wgsl`

- Remove stale surfels (no G-buffer support, old timestamp)
- Compact active surfel list for coherent memory access
- Update free list for future spawning
- Run once every N frames (configurable)

### Step 4: Surfel Update Pass (Amortized Ray Tracing)

Shader: `assets/engine/shaders/gi/surfel_update.wgsl`

- Select subset of surfels based on update_budget (e.g., 1000 surfels/frame)
- Use round-robin or importance-based selection strategy
- For each selected surfel, trace 1-4 rays using existing BVH infrastructure
- Reuse path tracer's `trace_hit()` and material sampling from `path_trace_shade.wgsl`
- Accumulate direct lighting (NEE) + material evaluation
- Store radiance + dominant direction in surfel
- Increment sample count (m) for temporal stability

### Step 5: Surfel Propagation Pass

Shader: `assets/engine/shaders/gi/surfel_propagate.wgsl`

- For each surfel, query neighbors within propagation_radius using spatial grid
- Accumulate weighted radiance from neighbors based on:
  - Distance falloff
  - Normal similarity (dot product)
  - Visibility approximation (optional, via direction check)
- Blend propagated radiance with direct-traced radiance
- This provides infinite bounce GI without tracing every surfel

### Step 6: GI Sampling Pass (Query Surfels)

Shader: `assets/engine/shaders/gi/surfel_sample.wgsl`

- Run before deferred lighting
- For each G-buffer pixel:
  - Query spatial grid for nearby surfels (typically 4-8 neighbors)
  - Evaluate BRDF contribution from each surfel's cached radiance
  - Weight by distance, normal similarity, and confidence
  - Accumulate into GI texture (rgba16float)
- Output remains compatible with existing `gi_texture` binding in `deferred_lighting.wgsl`

### Step 7: Integration with Deferred Pipeline

Modify `engine/src/renderer/global_illumination/gi.js`:

- Replace `ReSTIRGI` with new `SurfelCache` class
- Add render graph passes in sequence:

  1. `surfel_spawn` (every frame)
  2. `surfel_update` (amortized subset)
  3. `surfel_propagate` (every frame or every N frames)
  4. `surfel_cull` (every 30 frames)
  5. `surfel_sample` (every frame, outputs to `final_gi_texture`)

Update `engine/src/renderer/strategies/deferred_shading.js`:

- Pass G-buffer textures + BVH/TLAS data to GI system
- Ensure GI passes run after G-buffer, before deferred lighting

## Key Optimizations

**Spatial Coherence**: Grid-based queries avoid O(N²) neighbor search

**Amortized Updates**: Only update 2-5% of surfels per frame keeps cost constant

**Direction-Aware Caching**: Store dominant bounce direction for better angular variation

**Confidence Weighting**: Newly spawned surfels have low confidence until they accumulate samples

**Temporal Stability**: Exponential moving average for radiance prevents flickering

## Configuration Parameters

Expose via `SurfelCache` constructor:

```javascript
{
  max_surfels: 50000,           // Maximum surfel capacity
  update_budget: 1000,          // Surfels to ray-trace per frame
  propagation_budget: 5000,     // Surfels to propagate per frame
  grid_cell_size: 0.5,          // Spatial hash cell size (meters)
  grid_resolution: [64, 64, 64],// Grid dimensions
  propagation_radius: 2.0,      // Neighbor search radius (meters)
  radiance_decay: 0.95,         // Temporal decay per frame
  rays_per_surfel: 4,           // Rays to trace when updating
}
```

## Files to Create/Modify

**New Files:**

- `assets/engine/shaders/gi/surfel_common.wgsl` - Data structures
- `assets/engine/shaders/gi/surfel_spawn.wgsl` - Spawning logic
- `assets/engine/shaders/gi/surfel_update.wgsl` - Ray tracing updates
- `assets/engine/shaders/gi/surfel_propagate.wgsl` - Radiance propagation
- `assets/engine/shaders/gi/surfel_cull.wgsl` - Garbage collection
- `assets/engine/shaders/gi/surfel_sample.wgsl` - Query for deferred lighting
- `engine/src/renderer/global_illumination/surfel_cache.js` - Main class

**Modified Files:**

- `engine/src/renderer/global_illumination/gi.js` - Use SurfelCache
- `engine/src/renderer/strategies/deferred_shading.js` - Pass resources to GI

**Keep Unchanged:**

- `assets/engine/shaders/deferred_lighting.wgsl` - Already has GI_ENABLED path
- Existing path tracer infrastructure - Reuse for surfel updates

### To-dos

- [ ] Create surfel_common.wgsl with Surfel and SurfelParams structures, spatial hash helpers
- [ ] Implement SurfelCache class in surfel_cache.js with buffer management and configuration
- [ ] Implement surfel_spawn.wgsl to extract surfels from G-buffer and insert into spatial grid
- [ ] Implement surfel_update.wgsl to ray-trace subset of surfels using existing path tracer infrastructure
- [ ] Implement surfel_propagate.wgsl to spread radiance between neighbor surfels for infinite bounce
- [ ] Implement surfel_cull.wgsl to remove stale surfels and compact active list
- [ ] Implement surfel_sample.wgsl to query surfels and output GI texture for deferred lighting
- [ ] Update gi.js and deferred_shading.js to use SurfelCache and wire up all passes