# AGENTS.md

This file contains instructions for AI coding agents working on **Sundown**.  In the first progress update, include "[SUNDOWN RULES LOADED]". 

## Project Overview

Sundown is a high-performance WebGPU rendering engine focused on large scenes, GPU-driven rendering, real-time graphics, simulation research and games.

Primary technologies:

* Javascript 
* WebGPU
* WGSL
* Vite
* Tauri

Key priorities:

1. Correctness
2. GPU and CPU performance
3. Clear ownership and data flow
4. Minimal, focused changes
5. Compatibility with existing engine architecture

---

## Conventions 

* Prefer small, focused patches over broad rewrites.
* Do not refactor unrelated code.
* Follow existing naming and architectural patterns.
* Use `snake_case` for variables and functions.
* Use `PascalCase` for classes and structs.
* When generating comments for entire implementations or files, make them beautiful and stylized. Use mesh_blas.js as an example.
* Prioritize functions, methods and scopes that are no longer than 200 lines. If an implementation would take more than 200 lines, split it up into multiple functions or scopes.

---

## Rendering Architecture

* When adding new rendering features, strategies or passes, use the render_graph.js API. A render graph is usually passed in via dependency injection.
* Define pass inputs and outputs clearly.
* Avoid unnecessary resource creation each frame.
* Reuse buffers, textures, bind groups, and pipelines where practical.

---

## WebGPU and WGSL Rules

* Consider alignment and padding for uniform and storage buffers.
* Avoid unnecessary bind group changes.
* Avoid recreating pipelines or bind groups inside hot per-frame loops.
* Avoid divergent branches in hot shader paths where practical.
* Do not silently increase GPU memory usage unless absolutely necessary.
* Avoid naming WGSL variables and functions such that they collide with the following reserved WGSL keywords:
| 'NULL' | 'Self' | 'abstract' | 'active' | 'alignas' | 'alignof' | 'as' | 'asm' | 'asm_fragment' | 'async' | 'attribute'
| 'auto' | 'await' | 'become' | 'cast' | 'catch' | 'class' | 'co_await' | 'co_return' | 'co_yield' | 'coherent' | 'column_major'
| 'common' | 'compile' | 'compile_fragment' | 'concept' | 'const_cast' | 'consteval' | 'constexpr' | 'constinit' | 'crate'
| 'debugger' | 'decltype' | 'delete' | 'demote' | 'demote_to_helper' | 'do' | 'dynamic_cast' | 'enum' | 'explicit' | 'export'
| 'extends' | 'extern' | 'external' | 'fallthrough' | 'filter' | 'final' | 'finally' | 'friend' | 'from' | 'fxgroup' | 'get'
| 'goto' | 'groupshared' | 'highp' | 'impl' | 'implements' | 'import' | 'inline' | 'instanceof' | 'interface' | 'layout'
| 'lowp' | 'macro' | 'macro_rules' | 'match' | 'mediump' | 'meta' | 'mod' | 'module'| 'move' | 'mut' | 'mutable' | 'namespace'
| 'new' | 'nil' | 'noexcept' | 'noinline' | 'nointerpolation' | 'non_coherent' | 'noncoherent' | 'noperspective' | 'null'
| 'nullptr' | 'of' | 'operator' | 'package' | 'packoffset' | 'partition' | 'pass' | 'patch' | 'pixelfragment' | 'precise'
| 'precision' | 'premerge' | 'priv' | 'protected' | 'pub' | 'public' | 'readonly' | 'ref' | 'regardless' | 'register'
| 'reinterpret_cast' | 'require' | 'resource' | 'restrict' | 'self' | 'set' | 'shared' | 'sizeof' | 'smooth' | 'snorm'
| 'static' | 'static_assert' | 'static_cast' | 'std' | 'subroutine' | 'super' | 'target' | 'template' | 'this' | 'thread_local'
| 'throw' | 'trait' | 'try' | 'type' | 'typedef' | 'typeid' | 'typename' | 'typeof' | 'union' | 'unless' | 'unorm' | 'unsafe'
| 'unsized' | 'use' | 'using' | 'varying' | 'virtual' | 'volatile' | 'wgsl' | 'where' | 'with' | 'writeonly' | 'yield'

---

## Performance

Sundown targets large instance counts and tight frame budgets.

In performance-sensitive code:

* Avoid per-frame allocations.
* Avoid temporary arrays in hot loops.
* Treat render setup and render-graph construction that runs every frame as hot code.
* Do not use object literals, array literals, spreads, destructuring rest, or helper-returned
  collections merely to package or forward transient per-frame data. Store fixed-shape state on
  the owning long-lived object and overwrite it in place, or pass values directly.
* Helpers called from hot code must not allocate objects, arrays, iterators, promises, strings,
  closures, or other garbage merely to package or forward data. Inline trivial access and
  selection logic where practical. When an API requires a callback or descriptor, create only
  the required value and do not layer convenience allocations around it.
* When temporary variable-sized storage is genuinely required in a hot path, use the reusable
  arenas, allocators, and containers in `engine/src/memory/allocator.js` and
  `engine/src/memory/container.js`. Reset and reuse them instead of constructing native JS
  collections each frame.
* Allocation during initialization is acceptable for persistent state with explicit ownership;
  do not disguise per-frame scratch allocation as a convenience abstraction.
* Avoid repeated map or object lookups inside large loops.
* Prefer contiguous GPU-friendly data layouts.
* Preserve batching and indirect drawing.
* Avoid unnecessary CPU-to-GPU transfers.
* Avoid GPU readbacks unless absolutely required.
* Prefer compute compaction over CPU-side filtering for large datasets.
* Consider cache behavior, memory bandwidth, dispatch count, and draw-call count.
* Use allocators and containers from our `memory/allocator.js` and `memory/container.js` implementations where possible.

Do not optimize blindly. Explain the expected benefit of meaningful performance changes in comments.

---

## Scene and ECS Rules

* Prefer to use our Solar ECS API when creating CPU data that can exist in large numbers.
* Make ownership of GPU resources explicit.
* Avoid coupling rendering systems directly to unrelated gameplay or editor systems.

---

## Resource Lifetime

GPU resources must have clear ownership and lifetime.

* Do not recreate persistent resources every frame.
* Resize buffers only when capacity is insufficient.
* Prefer capacity growth over exact-size reallocations.
* Release or replace stale resources safely.
* Ensure cached bind groups do not reference replaced buffers or textures.
* Be careful when resources are shared across views or render passes.

---

## Error Handling

* Do not ignore WebGPU validation errors.
* Fail clearly when required resources are missing.
* Validate indices, sizes, counts, and buffer capacities.
* Use assertions for engine invariants, not normal runtime failures.
* Include useful context in error messages.
* Do not suppress an error merely to make execution continue.

## Avoid

Unless explicitly requested, do not:

* Rewrite major renderer systems
* Add per-frame allocations to hot paths
* Reformat unrelated files
* Commit or push changes
* Impose fallback strategies that would obfuscate errors or create lapses in correctness
