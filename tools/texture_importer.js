import fs from "fs/promises";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { Worker, isMainThread, parentPort, workerData } from "worker_threads";
import sharp from "sharp";
import { encodeToKTX2 } from "ktx2-encoder";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const project_root = path.resolve(__dirname, "..");
const asset_root = path.join(project_root, "assets");

const texture_extensions = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".tif",
  ".tiff",
  ".tga",
  ".bmp",
  ".webp",
]);

// Browser UI images are consumed by HTML Image/Canvas APIs, and generated font
// atlases are disposable build artifacts. Neither belongs to the GPU texture import.
const excluded_asset_directories = [
  path.join(asset_root, "engine", "sprites"),
  path.join(asset_root, "engine", "fonts"),
  path.join(asset_root, "cooked"),
];

const pool_dimension_caps = Object.freeze({
  albedo: 1024,
  normal: 1024,
  roughness: 512,
  metallic: 512,
  ao: 512,
  height: 512,
  specular: 256,
  emission: 256,
});

const ktx2_identifier = new Uint8Array([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

function is_within(root, file_path) {
  const relative = path.relative(root, file_path);
  return relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative);
}

function is_excluded_asset_path(file_path) {
  return excluded_asset_directories.some(
    (directory) => file_path === directory || is_within(directory, file_path)
  );
}

function to_asset_path(file_path) {
  return path.relative(asset_root, file_path).split(path.sep).join("/");
}

function replace_extension(file_path, extension = ".ktx2") {
  return file_path.slice(0, -path.extname(file_path).length) + extension;
}

function infer_texture_usage(file_path) {
  const name = path.basename(file_path, path.extname(file_path)).toLowerCase();
  const normalized_path = file_path.split(path.sep).join("/").toLowerCase();
  if (/(skybox|gradientbox)/.test(normalized_path)) return "color";
  if (/(albedo|diffuse|base[_ -]?color|emissi|wallpaper|carpet|ceiling)/.test(name)) {
    return "color";
  }
  return "data";
}

function infer_pool_key(file_path) {
  const name = path.basename(file_path, path.extname(file_path)).toLowerCase();
  if (/(normal|nrm)/.test(name)) return "normal";
  if (/(roughness|rough)/.test(name)) return "roughness";
  if (/(metallic|metalness)/.test(name)) return "metallic";
  if (/(ambient[_ -]?occlusion|[_ -]ao|^ao[_ -])/.test(name)) return "ao";
  if (/(height|displace)/.test(name)) return "height";
  if (/specular/.test(name)) return "specular";
  if (/(emission|emissive)/.test(name)) return "emission";
  if (/(albedo|diffuse|base[_ -]?color|wallpaper|carpet|ceiling)/.test(name)) return "albedo";
  return null;
}

function choose_encoding(source_path, metadata, requested_encoding) {
  if (requested_encoding) return requested_encoding;
  const normalized_path = source_path.split(path.sep).join("/").toLowerCase();
  if (
    metadata.pool_key === "normal" ||
    metadata.pool_key === "height" ||
    normalized_path.includes("/textures/noise/")
  ) {
    return "uastc";
  }
  return "etc1s";
}

function texture_source_index(texture) {
  return texture?.extensions?.KHR_texture_basisu?.source ?? texture?.source;
}

function register_gltf_texture_usage(context, gltf_path, texture_index, pool_key) {
  if (!Number.isInteger(texture_index)) return;
  const texture = context.gltf.textures?.[texture_index];
  const source_index = texture_source_index(texture);
  const image = Number.isInteger(source_index) ? context.gltf.images?.[source_index] : null;
  if (!image?.uri || image.uri.startsWith("data:")) return;

  const source_path = path.resolve(path.dirname(gltf_path), decodeURIComponent(image.uri));
  if (!texture_extensions.has(path.extname(source_path).toLowerCase())) return;

  const cap = pool_dimension_caps[pool_key];
  const usage = pool_key === "albedo" || pool_key === "emission" ? "color" : "data";
  const existing = context.sources.get(source_path);

  if (!existing || cap > existing.dimension_cap) {
    context.sources.set(source_path, {
      usage,
      pool_key,
      dimension_cap: cap,
    });
  } else if (usage === "color") {
    existing.usage = "color";
  }
}

async function collect_files(root, predicate, options = {}) {
  const files = [];

  async function visit(directory) {
    if (options.skip_directory?.(directory)) return;
    let entries;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if (error.code === "ENOENT") return;
      throw error;
    }

    for (const entry of entries) {
      const file_path = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(file_path);
      else if (predicate(file_path)) files.push(file_path);
    }
  }

  await visit(root);
  return files;
}

async function collect_gltf_context() {
  const sources = new Map();
  const gltf_files = await collect_files(
    asset_root,
    (file_path) => path.extname(file_path).toLowerCase() === ".gltf"
  );

  for (const gltf_path of gltf_files) {
    const gltf = JSON.parse(await fs.readFile(gltf_path, "utf8"));
    const context = { gltf, sources };

    for (const material of gltf.materials ?? []) {
      const pbr = material.pbrMetallicRoughness;
      register_gltf_texture_usage(context, gltf_path, pbr?.baseColorTexture?.index, "albedo");
      register_gltf_texture_usage(context, gltf_path, material.normalTexture?.index, "normal");
      register_gltf_texture_usage(
        context,
        gltf_path,
        pbr?.metallicRoughnessTexture?.index,
        "roughness"
      );
      register_gltf_texture_usage(context, gltf_path, material.occlusionTexture?.index, "ao");
      register_gltf_texture_usage(context, gltf_path, material.emissiveTexture?.index, "emission");

      const specular = material.extensions?.KHR_materials_specular;
      register_gltf_texture_usage(context, gltf_path, specular?.specularTexture?.index, "specular");
      register_gltf_texture_usage(
        context,
        gltf_path,
        specular?.specularColorTexture?.index,
        "albedo"
      );
    }
  }

  return { sources, gltf_files };
}

async function collect_texture_sources(positional_paths) {
  if (positional_paths.length > 0) {
    const sources = [];
    for (const input_path of positional_paths) {
      const source_path = path.resolve(input_path);
      if (!is_within(asset_root, source_path)) {
        throw new Error(`Texture import source must be inside '${asset_root}': '${source_path}'.`);
      }
      if (is_excluded_asset_path(source_path)) {
        throw new Error(
          `'${source_path}' is a browser UI or generated font asset, not a GPU texture.`
        );
      }
      if (!texture_extensions.has(path.extname(source_path).toLowerCase())) {
        throw new Error(`Unsupported texture source '${source_path}'.`);
      }
      await fs.access(source_path);
      sources.push(source_path);
    }
    return sources;
  }

  return collect_files(
    asset_root,
    (file_path) => texture_extensions.has(path.extname(file_path).toLowerCase()),
    { skip_directory: is_excluded_asset_path }
  );
}

async function decode_image(image_buffer) {
  const { data, info } = await sharp(image_buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    width: info.width,
    height: info.height,
    data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  };
}

async function prepare_source(source_path, dimension_cap) {
  const image = sharp(source_path).ensureAlpha();
  const metadata = await image.metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Texture '${source_path}' does not have valid dimensions.`);
  }

  if (!dimension_cap) {
    return {
      buffer: await fs.readFile(source_path),
      width: metadata.width,
      height: metadata.height,
    };
  }

  // Semantic pools require identical extents and mip counts for every compressed
  // array layer. Normalize once during import instead of resampling at runtime.
  const buffer = await image
    .resize(dimension_cap, dimension_cap, { fit: "fill", kernel: sharp.kernel.lanczos3 })
    .png()
    .toBuffer();
  return { buffer, width: dimension_cap, height: dimension_cap };
}

function validate_ktx2(encoded, output_path) {
  if (encoded.byteLength <= ktx2_identifier.byteLength) {
    throw new Error(`KTX2 encoder produced an empty payload for '${output_path}'.`);
  }
  for (let i = 0; i < ktx2_identifier.length; i++) {
    if (encoded[i] !== ktx2_identifier[i]) {
      throw new Error(`KTX2 encoder produced an invalid identifier for '${output_path}'.`);
    }
  }
}

async function import_texture(task) {
  const prepared = await prepare_source(task.source_path, task.dimension_cap);
  const perceptual = task.usage === "color";
  const encoded = await encodeToKTX2(new Uint8Array(prepared.buffer), {
    imageDecoder: decode_image,
    isUASTC: task.encoding === "uastc",
    isKTX2File: true,
    generateMipmap: task.generate_mips,
    needSupercompression: task.encoding === "uastc",
    enableRDO: task.encoding === "uastc",
    rdoQualityLevel: task.rdo_quality,
    uastcLDRQualityLevel: task.quality,
    qualityLevel: task.etc1s_quality,
    compressionLevel: task.compression_level,
    isPerceptual: perceptual,
    isSetKTX2SRGBTransferFunc: perceptual,
    enableDebug: false,
    kvData: {
      "sundown.texture_usage": task.usage,
      "sundown.texture_pool": task.pool_key ?? "default",
    },
  });

  validate_ktx2(encoded, task.output_path);
  const temp_path = `${task.output_path}.${process.pid}.importing`;
  try {
    await fs.writeFile(temp_path, encoded);
    await fs.rm(task.output_path, { force: true });
    await fs.rename(temp_path, task.output_path);
  } finally {
    await fs.rm(temp_path, { force: true });
  }

  const source_stat = await fs.stat(task.source_path);
  return {
    source_path: task.source_path,
    output_path: task.output_path,
    source_bytes: source_stat.size,
    imported_bytes: encoded.byteLength,
    encoding: task.encoding,
    width: prepared.width,
    height: prepared.height,
  };
}

function run_worker_pool(tasks, job_count, on_complete) {
  if (tasks.length === 0) return Promise.resolve([]);

  return new Promise((resolve, reject) => {
    const results = [];
    const workers = [];
    let next_task = 0;
    let completed = 0;
    let settled = false;

    function stop_workers() {
      return Promise.all(workers.map((worker) => worker.terminate()));
    }

    function fail(error) {
      if (settled) return;
      settled = true;
      void stop_workers().finally(() => reject(error));
    }

    function dispatch(worker) {
      if (next_task >= tasks.length) return;
      worker.postMessage(tasks[next_task++]);
    }

    const worker_count = Math.min(job_count, tasks.length);
    for (let i = 0; i < worker_count; i++) {
      const worker = new Worker(__filename, { workerData: { texture_import_worker: true } });
      workers.push(worker);
      worker.on("error", fail);
      worker.on("message", (message) => {
        if (settled) return;
        if (message.error) {
          fail(new Error(message.error));
          return;
        }

        results.push(message.result);
        completed++;
        on_complete?.(message.result, completed, tasks.length);
        if (completed === tasks.length) {
          settled = true;
          void stop_workers().then(() => resolve(results), reject);
        } else {
          dispatch(worker);
        }
      });
      dispatch(worker);
    }
  });
}

async function rewrite_gltf_references(gltf_files, source_to_output) {
  const modified_files = [];

  for (const gltf_path of gltf_files) {
    let text = await fs.readFile(gltf_path, "utf8");
    const gltf = JSON.parse(text);
    let changed = false;

    for (const image of gltf.images ?? []) {
      if (!image.uri || image.uri.startsWith("data:")) continue;
      const source_path = path.resolve(path.dirname(gltf_path), decodeURIComponent(image.uri));
      if (!source_to_output.has(source_path)) continue;

      const imported_uri = replace_extension(image.uri);
      const old_uri_json = JSON.stringify(image.uri);
      const new_uri_json = JSON.stringify(imported_uri);
      text = text.split(old_uri_json).join(new_uri_json);

      if (
        typeof image.name === "string" &&
        texture_extensions.has(path.extname(image.name).toLowerCase())
      ) {
        text = text
          .split(JSON.stringify(image.name))
          .join(JSON.stringify(replace_extension(image.name)));
      }
      changed = true;
    }

    if (changed) {
      await fs.writeFile(gltf_path, text);
      modified_files.push(gltf_path);
    }
  }

  return modified_files;
}

async function collect_reference_files() {
  const text_extensions = new Set([".js", ".mjs", ".json", ".html", ".css", ".md"]);
  const skipped_directories = new Set([".git", "node_modules", "dist", "out"]);
  return collect_files(
    project_root,
    (file_path) => {
      const extension = path.extname(file_path).toLowerCase();
      return extension !== ".gltf" && text_extensions.has(extension);
    },
    {
      skip_directory(directory) {
        return (
          skipped_directories.has(path.basename(directory)) || is_excluded_asset_path(directory)
        );
      },
    }
  );
}

async function rewrite_asset_references(reference_files, source_to_output) {
  const replacements = [];
  for (const [source_path, output_path] of source_to_output) {
    replacements.push([to_asset_path(source_path), to_asset_path(output_path)]);
  }

  const modified_files = [];
  for (const file_path of reference_files) {
    let text = await fs.readFile(file_path, "utf8");
    const original = text;
    for (const [source_url, output_url] of replacements) {
      text = text.split(source_url).join(output_url);
      text = text.split(encodeURI(source_url)).join(encodeURI(output_url));
    }
    if (text !== original) {
      await fs.writeFile(file_path, text);
      modified_files.push(file_path);
    }
  }
  return modified_files;
}

async function verify_references(gltf_files, reference_files, source_to_output) {
  const stale_references = [];

  for (const gltf_path of gltf_files) {
    const gltf = JSON.parse(await fs.readFile(gltf_path, "utf8"));
    for (const image of gltf.images ?? []) {
      if (!image.uri || image.uri.startsWith("data:")) continue;
      const source_path = path.resolve(path.dirname(gltf_path), decodeURIComponent(image.uri));
      if (source_to_output.has(source_path)) {
        stale_references.push(`${gltf_path}: ${image.uri}`);
      }
    }
  }

  const source_urls = [...source_to_output.keys()].map(to_asset_path);
  for (const file_path of reference_files) {
    const text = await fs.readFile(file_path, "utf8");
    for (const source_url of source_urls) {
      if (text.includes(source_url) || text.includes(encodeURI(source_url))) {
        stale_references.push(`${file_path}: ${source_url}`);
      }
    }
  }

  if (stale_references.length > 0) {
    throw new Error(
      `Refusing to delete source textures; stale references remain:\n${stale_references.join("\n")}`
    );
  }
}

function parse_args(argv) {
  const options = {
    apply: false,
    generate_mips: true,
    jobs: Math.max(1, Math.min(2, os.availableParallelism() - 1)),
    rdo_quality: 1.0,
    quality: 2,
    etc1s_quality: 255,
    compression_level: 2,
  };
  const positional = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--apply") options.apply = true;
    else if (arg === "--no-mips") options.generate_mips = false;
    else if (arg === "--encoding") options.encoding = argv[++i];
    else if (arg === "--max-dimension") options.dimension_cap = Number(argv[++i]);
    else if (arg === "--jobs") options.jobs = Number(argv[++i]);
    else if (arg === "--rdo-quality") options.rdo_quality = Number(argv[++i]);
    else if (arg === "--quality") options.quality = Number(argv[++i]);
    else if (arg === "--etc1s-quality") options.etc1s_quality = Number(argv[++i]);
    else positional.push(arg);
  }

  if (!Number.isInteger(options.jobs) || options.jobs < 1) {
    throw new Error("--jobs must be a positive integer.");
  }
  if (options.encoding && options.encoding !== "uastc" && options.encoding !== "etc1s") {
    throw new Error("--encoding must be 'uastc' or 'etc1s'.");
  }
  return { options, positional };
}

function format_mib(bytes) {
  return (bytes / (1024 * 1024)).toFixed(2);
}

async function run_import() {
  const { options, positional } = parse_args(process.argv.slice(2));
  const { sources: gltf_usages, gltf_files } = await collect_gltf_context();
  const source_paths = await collect_texture_sources(positional);
  const source_to_output = new Map();
  const output_to_source = new Map();
  const tasks = [];
  let source_bytes = 0;

  for (const source_path of source_paths) {
    const output_path = replace_extension(source_path);
    const collision = output_to_source.get(output_path);
    if (collision && collision !== source_path) {
      throw new Error(`Texture import collision: '${collision}' and '${source_path}'.`);
    }
    output_to_source.set(output_path, source_path);
    source_to_output.set(source_path, output_path);

    const source_stat = await fs.stat(source_path);
    source_bytes += source_stat.size;
    const pool_key = gltf_usages.get(source_path)?.pool_key ?? infer_pool_key(source_path);
    const metadata = gltf_usages.get(source_path) ?? {
      usage: infer_texture_usage(source_path),
      pool_key,
      dimension_cap: pool_key ? pool_dimension_caps[pool_key] : undefined,
    };

    tasks.push({
      source_path,
      output_path,
      usage: metadata.usage,
      pool_key: metadata.pool_key,
      dimension_cap: options.dimension_cap ?? metadata.dimension_cap,
      encoding: choose_encoding(source_path, metadata, options.encoding),
      generate_mips: options.generate_mips,
      rdo_quality: options.rdo_quality,
      quality: options.quality,
      etc1s_quality: options.etc1s_quality,
      compression_level: options.compression_level,
    });
  }

  const uastc_count = tasks.filter((task) => task.encoding === "uastc").length;
  const etc1s_count = tasks.length - uastc_count;
  console.log(
    `[texture_importer] ${tasks.length} source textures (${format_mib(source_bytes)} MiB): ` +
      `${uastc_count} UASTC, ${etc1s_count} ETC1S`
  );

  if (tasks.length === 0) {
    console.log("[texture_importer] Nothing to import.");
    return;
  }
  if (!options.apply) {
    console.log(
      "[texture_importer] Dry run only. Re-run with --apply to convert, rewrite references, and delete the sources."
    );
    return;
  }

  const results = await run_worker_pool(tasks, options.jobs, (result, completed, total) => {
    console.log(
      `[texture_importer] ${completed}/${total} ${to_asset_path(result.source_path)} -> ` +
        `${path.basename(result.output_path)} (${result.encoding})`
    );
  });

  const reference_files = await collect_reference_files();
  const modified_gltf_files = await rewrite_gltf_references(gltf_files, source_to_output);
  const modified_reference_files = await rewrite_asset_references(
    reference_files,
    source_to_output
  );
  await verify_references(gltf_files, reference_files, source_to_output);

  for (const source_path of source_paths) {
    await fs.rm(source_path);
  }

  const imported_bytes = results.reduce((sum, result) => sum + result.imported_bytes, 0);
  console.log(
    `[texture_importer] Imported and removed ${results.length} sources: ` +
      `${format_mib(source_bytes)} MiB -> ${format_mib(imported_bytes)} MiB KTX2. ` +
      `Updated ${modified_gltf_files.length} glTF and ${modified_reference_files.length} text files.`
  );
}

if (!isMainThread && workerData?.texture_import_worker) {
  parentPort.on("message", async (task) => {
    try {
      parentPort.postMessage({ result: await import_texture(task) });
    } catch (error) {
      parentPort.postMessage({ error: error?.stack ?? String(error) });
    }
  });
} else {
  await run_import();
}
