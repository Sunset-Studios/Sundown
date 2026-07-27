// Compatibility exports for callers outside the engine. New code should import
// shared transport and serialization helpers from engine/src/streaming.
export {
  read_file,
  read_file_async,
  read_file_bytes_async,
} from "../streaming/streaming_io.js";
