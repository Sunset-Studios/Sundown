import {
  read_binary_manifest_async,
  resolve_manifest_asset_path,
} from "../streaming/streaming_io.js";

const sbvh_sidecar_promise_cache = new Map();

export function load_sbvh_sidecar_async(gltf_path) {
  if (!gltf_path || !gltf_path.toLowerCase().endsWith(".gltf")) {
    return Promise.resolve(null);
  }

  if (sbvh_sidecar_promise_cache.has(gltf_path)) {
    return sbvh_sidecar_promise_cache.get(gltf_path);
  }

  const sidecar_promise = (async () => {
    const manifest_path = gltf_path.replace(/\.gltf$/i, ".sbvh.json");
    try {
      return await read_binary_manifest_async(manifest_path, {
        label: "SBVH sidecar",
        optional: true,
        resolve_binary_path: (manifest) =>
          manifest.binary
            ? resolve_manifest_asset_path(manifest_path, manifest.binary)
            : gltf_path.replace(/\.gltf$/i, ".sbvh.bin"),
      });
    } catch {
      return null;
    }
  })();

  sbvh_sidecar_promise_cache.set(gltf_path, sidecar_promise);
  return sidecar_promise;
}

export function get_cooked_sbvh_for_mesh(sidecar, mesh_index) {
  if (!sidecar || mesh_index < 0) {
    return null;
  }

  const manifest_mesh = sidecar.manifest.meshes?.[mesh_index];
  if (!manifest_mesh || manifest_mesh.skipped) {
    return null;
  }

  const section = sidecar.manifest.sections?.nodes;
  if (!section) {
    return null;
  }

  const node_offset = manifest_mesh.nodeOffset ?? 0;
  const node_count = manifest_mesh.nodeCount ?? 0;
  if (node_count <= 0) {
    return null;
  }

  const float_offset = (section.offset / Float32Array.BYTES_PER_ELEMENT) + node_offset * 8;
  const node_data = new Float32Array(sidecar.binary, float_offset * Float32Array.BYTES_PER_ELEMENT, node_count * 8);

  return {
    primitive_count: manifest_mesh.primitiveCount >>> 0,
    reference_count: manifest_mesh.referenceCount >>> 0,
    node_count: node_count >>> 0,
    node_data: new Float32Array(node_data),
  };
}
