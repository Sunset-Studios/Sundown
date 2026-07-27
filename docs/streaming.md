# Streaming and serialization

Sundown's common streaming layer lives in `engine/src/streaming`. It separates
frame scheduling from domain-specific work:

- `StreamingSystem` owns provider registration, request state, cancellation,
  completion, and per-frame dispatch.
- `StreamProvider` defines the hooks for incremental streaming and
  serialization.
- `streaming_io.js` owns shared text, byte, JSON, and manifest-plus-binary
  resource loading.
- `TextureStreamingProvider` is the first concrete implementation. Texture
  decoding, mip generation, worker submission, and bounded GPU upload now live
  under `streaming/providers` instead of `renderer/texture.js`.
- Shader and texture worker load jobs live beside streaming providers; their
  former utility paths remain as compatibility exports.

Binary layouts remain with their domains. Shader archives, meshlet sidecars,
and SBVH sidecars interpret their own records, but use the common I/O layer to
acquire and deserialize their manifest/binary bundles.

## Registering a provider

The simulator installs the common system as a simulation layer. A custom
provider can be registered directly or passed through
`Simulator.create(..., { streaming: { providers: [...] } })` during application
setup:

```js
import {
  StreamProvider,
  StreamUpdateStatus,
  StreamingSystem,
} from "./engine/src/streaming/index.js";

class GameplayChunkProvider extends StreamProvider {
  static provider_type = "gameplay_chunk";

  begin_stream(request) {
    return {
      records: request.options.records,
      next_record: 0,
    };
  }

  begin_frame() {
    return { records_remaining: 64 };
  }

  update_stream(request, { frame }) {
    const state = request.state;

    while (state.next_record < state.records.length && frame.records_remaining > 0) {
      request.target.add(state.records[state.next_record++]);
      frame.records_remaining--;
    }

    return state.next_record === state.records.length
      ? StreamUpdateStatus.COMPLETE
      : StreamUpdateStatus.CONTINUE;
  }

  cancel_stream(request) {
    request.target.rollback?.();
  }

  serialize(chunk) {
    return JSON.stringify(chunk);
  }

  deserialize(payload) {
    return JSON.parse(payload);
  }
}

StreamingSystem.register_provider(GameplayChunkProvider);

const request = StreamingSystem.stream("gameplay_chunk", world_chunk, {
  records,
});
await request.finished;

if (request.error) {
  console.error(request.error);
}
```

Provider-specific constructor options can be supplied to the simulator with a
descriptor:

```js
await Simulator.create("gpu_canvas", null, {
  streaming: {
    providers: [
      {
        provider: GameplayChunkProvider,
        options: { records_per_frame: 64 },
      },
    ],
  },
});
```

`begin_stream()` may return provider state immediately or return a promise.
`update_stream()` is intentionally synchronous: it runs inside the simulation
frame and should spend only the budget created by `begin_frame()`.

## Provider lifecycle

1. `begin_stream(request)` starts acquisition and returns provider state.
2. `begin_frame(context)` creates a budget shared by that provider's active
   requests for the current frame.
3. `update_stream(request, context)` performs bounded work and returns a
   `StreamUpdateStatus`.
4. `complete_stream(request)` publishes or finalizes the streamed resource.
5. `cancel_stream(request)` releases partial work on cancellation or failure.
6. `serialize(value)` and `deserialize(payload)` provide the provider's
   persistence boundary through `StreamingSystem.serialize()` and
   `StreamingSystem.deserialize()`.

`request.finished` always resolves with the request. Inspect `request.status`,
`request.result`, and `request.error` for its terminal outcome.
