import { BasisUniversal, TranscoderTextureFormat } from "@h00w/basis-universal-transcoder";
import basis_wasm_url from "@h00w/basis-universal-transcoder/basis_capi_transcoder.wasm?url";

let basis_module_promise = null;

async function instantiate_basis_wasm(imports) {
  if (WebAssembly.instantiateStreaming) {
    try {
      return await WebAssembly.instantiateStreaming(fetch(basis_wasm_url), imports);
    } catch {
      // Some development servers do not label WASM correctly. Fetch a fresh body
      // for the ArrayBuffer fallback because instantiateStreaming consumes its body.
    }
  }
  const response = await fetch(basis_wasm_url);
  if (!response.ok) {
    throw new Error(
      `Failed to load the Basis Universal transcoder: ${response.status} ${response.statusText}`
    );
  }
  return WebAssembly.instantiate(await response.arrayBuffer(), imports);
}

function get_basis_module() {
  basis_module_promise ??= BasisUniversal.getInstance(instantiate_basis_wasm);
  return basis_module_promise;
}

function select_first_mip(width, height, levels, max_dimension) {
  if (!Number.isFinite(max_dimension)) return 0;
  let first_mip = 0;
  while (
    first_mip + 1 < levels &&
    Math.max(Math.max(1, width >> first_mip), Math.max(1, height >> first_mip)) > max_dimension
  ) {
    first_mip++;
  }
  return first_mip;
}

export async function transcode_ktx2(ktx2_bytes, options = {}) {
  const basis = await get_basis_module();
  const transcoder = basis.createKTX2Transcoder();

  try {
    if (!transcoder.init(ktx2_bytes)) {
      throw new Error("Basis Universal rejected the KTX2 payload.");
    }

    const header = transcoder.getHeader();
    const source_width = header.width;
    const source_height = header.height;
    const source_levels = header.levels;
    const source_layers = Math.max(1, header.layers);
    const source_faces = Math.max(1, header.faces);
    if (source_layers !== 1 || source_faces !== 1) {
      throw new Error(
        `Sundown V1 expects one 2D image per KTX2 file, received ${source_layers} layers and ${source_faces} faces.`
      );
    }
    if (!transcoder.startTranscoding()) {
      throw new Error("Basis Universal could not start KTX2 transcoding.");
    }

    const first_mip = select_first_mip(
      source_width,
      source_height,
      source_levels,
      options.max_dimension
    );
    const mip_levels = options.no_mips ? 1 : source_levels - first_mip;
    const target_format = options.supports_bc
      ? TranscoderTextureFormat.cTFBC7_RGBA
      : TranscoderTextureFormat.cTFRGBA32;
    const format = options.supports_bc ? "bc7-rgba-unorm" : "rgba8unorm";
    const mip_chain = [];

    for (let level = 0; level < mip_levels; level++) {
      const source_level = first_mip + level;
      const result = transcoder.transcodeImageLevel({
        format: target_format,
        level: source_level,
      });
      if (!result) {
        throw new Error(`Basis Universal failed to transcode KTX2 mip ${source_level}.`);
      }

      const data = result.data.slice();
      const blocks_per_row = Math.max(1, Math.ceil(result.width / 4));
      const block_rows = Math.max(1, Math.ceil(result.height / 4));
      mip_chain.push({
        data,
        width: result.width,
        height: result.height,
        copy_width: options.supports_bc ? blocks_per_row * 4 : result.width,
        copy_height: options.supports_bc ? block_rows * 4 : result.height,
        bytes_per_row: options.supports_bc ? blocks_per_row * 16 : result.width * 4,
        rows_per_image: options.supports_bc ? block_rows : result.height,
      });
    }

    return {
      mip_chain,
      source_width,
      source_height,
      width: mip_chain[0].width,
      height: mip_chain[0].height,
      mip_levels: mip_chain.length,
      format,
      compressed: options.supports_bc === true,
    };
  } finally {
    transcoder.dispose();
  }
}
