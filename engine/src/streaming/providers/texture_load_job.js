import {
  create_job_result,
  resolve_href,
  is_html_content_type,
} from "../../utility/job_worker_runtime.js";
import { transcode_ktx2 } from "../../renderer/ktx2_transcoder.js";

function resolve_ktx2_paths(resolved_paths) {
  if (!resolved_paths.every((url) => new URL(url).pathname.endsWith(".ktx2"))) return null;
  return { paths: resolved_paths };
}

function compute_mip_levels(width, height, no_mips = false) {
  return no_mips ? 1 : Math.floor(Math.log2(Math.max(width, height))) + 1;
}

function resolve_target_texture_config(source_width, source_height, payload) {
  const no_mips = !!payload?.no_mips;
  const pool_dimension_cap = payload?.pool_dimension_cap;
  const max_texture_dimension = payload?.max_texture_dimension;
  const dimension_cap = Math.min(
    Number.isFinite(pool_dimension_cap) ? pool_dimension_cap : Number.POSITIVE_INFINITY,
    Number.isFinite(max_texture_dimension) ? max_texture_dimension : Number.POSITIVE_INFINITY
  );
  const has_dimension_cap = Number.isFinite(dimension_cap);

  if (!has_dimension_cap) {
    return {
      width: source_width,
      height: source_height,
      mip_levels: compute_mip_levels(source_width, source_height, no_mips),
    };
  }

  const normalized_width = Math.max(1, Math.min(source_width, dimension_cap));
  const normalized_height = Math.max(1, Math.min(source_height, dimension_cap));
  const normalized_mip_levels = compute_mip_levels(normalized_width, normalized_height, no_mips);

  return {
    width: Math.max(normalized_width, payload?.pooled_width ?? 0),
    height: Math.max(normalized_height, payload?.pooled_height ?? 0),
    mip_levels: Math.max(normalized_mip_levels, payload?.pooled_mip_levels ?? 1),
  };
}

async function build_mip_chain(source, width, height, mip_levels, signal) {
  signal?.throwIfAborted?.();

  let mip0_source = source;
  if (source.width !== width || source.height !== height) {
    mip0_source = await createImageBitmap(source, {
      resizeWidth: width,
      resizeHeight: height,
      resizeQuality: "high",
    });
  }

  const mip_chain = [mip0_source];
  const transferables = [mip0_source];

  for (let lvl = 1; lvl < mip_levels; lvl++) {
    signal?.throwIfAborted?.();

    const mip_width = Math.max(1, width >> lvl);
    const mip_height = Math.max(1, height >> lvl);
    const mip_bitmap = await createImageBitmap(mip0_source, {
      resizeWidth: mip_width,
      resizeHeight: mip_height,
      resizeQuality: "high",
    });

    mip_chain.push(mip_bitmap);
    transferables.push(mip_bitmap);
  }

  if (mip0_source !== source) {
    source.close?.();
  }

  return {
    mip_chain,
    transferables,
  };
}

async function fetch_texture_bytes(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch texture '${url}': ${response.status} ${response.statusText}`);
  }
  const content_type = response.headers.get("content-type") || "";
  if (is_html_content_type(content_type)) {
    throw new Error(`Expected texture asset at '${url}', but received HTML`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function load_ktx2_texture_data(ktx2, original_paths, payload, signal, post_progress) {
  const mip_chains = [];
  const transferables = [];
  let base = null;

  for (let layer = 0; layer < ktx2.paths.length; layer++) {
    const url = ktx2.paths[layer];
    const bytes = await fetch_texture_bytes(url, signal);
    signal?.throwIfAborted?.();
    const transcoded = await transcode_ktx2(bytes, {
      supports_bc: payload.supports_bc === true,
      max_dimension: payload.max_texture_dimension,
      no_mips: payload.no_mips,
    });

    if (
      base &&
      (transcoded.width !== base.width ||
        transcoded.height !== base.height ||
        transcoded.mip_levels !== base.mip_levels ||
        transcoded.format !== base.format)
    ) {
      throw new Error(
        `KTX2 texture layers must have matching extents, mip counts, and formats ('${original_paths[layer]}').`
      );
    }
    base ??= transcoded;
    mip_chains.push(transcoded.mip_chain);
    for (const mip of transcoded.mip_chain) transferables.push(mip.data.buffer);

    post_progress?.({
      loaded: layer + 1,
      total: ktx2.paths.length,
      path: original_paths[layer],
      url,
      ktx2: true,
    });
  }

  return create_job_result(
    {
      mip_chains,
      resolved_paths: ktx2.paths,
      source_width: base.source_width,
      source_height: base.source_height,
      width: base.width,
      height: base.height,
      mip_levels: base.mip_levels,
      format: base.format,
      compressed: base.compressed,
      texture_usage: payload.texture_usage ?? "data",
    },
    transferables
  );
}

export async function load_texture_bitmaps_job(payload, { signal, post_progress }) {
  const paths = payload?.paths ?? [];
  const base_url = payload?.base_url ?? self.location.href;
  if (!Array.isArray(paths) || paths.length === 0) {
    throw new Error("Texture load job requires one or more paths");
  }

  const resolved_paths = paths.map((path) => resolve_href(path, base_url));
  const ktx2 = resolve_ktx2_paths(resolved_paths);
  if (ktx2) {
    return load_ktx2_texture_data(ktx2, paths, payload, signal, post_progress);
  }

  const decoded_bitmaps = [];
  const transferables = [];

  for (let i = 0; i < resolved_paths.length; i++) {
    const url = resolved_paths[i];
    const response = await fetch(url, { signal });
    if (!response.ok) {
      throw new Error(
        `Failed to fetch texture '${url}': ${response.status} ${response.statusText}`
      );
    }

    const content_type = response.headers.get("content-type") || "";
    if (is_html_content_type(content_type)) {
      throw new Error(`Expected image asset at '${url}', but received HTML`);
    }

    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob, {
      colorSpaceConversion: "none",
    });

    decoded_bitmaps.push(bitmap);

    post_progress?.({
      loaded: i + 1,
      total: resolved_paths.length,
      path: paths[i],
      url,
    });
  }

  const base_bitmap = decoded_bitmaps[0];
  const source_width = base_bitmap.width;
  const source_height = base_bitmap.height;
  const target_config = resolve_target_texture_config(source_width, source_height, payload);
  const mip_chains = [];

  for (let layer = 0; layer < decoded_bitmaps.length; layer++) {
    const { mip_chain, transferables: layer_transferables } = await build_mip_chain(
      decoded_bitmaps[layer],
      target_config.width,
      target_config.height,
      target_config.mip_levels,
      signal
    );
    mip_chains.push(mip_chain);
    transferables.push(...layer_transferables);
  }

  return create_job_result(
    {
      mip_chains,
      resolved_paths,
      source_width,
      source_height,
      width: target_config.width,
      height: target_config.height,
      mip_levels: target_config.mip_levels,
    },
    transferables
  );
}
