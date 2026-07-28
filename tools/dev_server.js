import { createServer as createViteServer } from "vite";
import express from "express";
import fs from "fs/promises";
import { createReadStream } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  SceneDataPackage,
  scene_data_package_fixed_header_byte_length,
} from "../engine/src/streaming/scene_data_package.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG_BASE_PATH = path.resolve(__dirname, "../assets/config");
const ASSETS_BASE_PATH = path.resolve(__dirname, "../assets");
const SCENE_UPLOAD_BASE_PATH = path.resolve(__dirname, "../out/.scene-uploads");

function normalize_project_root(project_root) {
  const normalized = String(project_root ?? "")
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "");
  if (!normalized) return [];

  const segments = normalized.split("/");
  if (
    segments.some(
      (segment) =>
        !segment || segment === "." || segment === ".." || !/^[a-zA-Z0-9._-]+$/.test(segment)
    )
  ) {
    throw new Error("Project root contains an unsafe path segment.");
  }
  return segments;
}

function sanitize_scene_name(scene_name) {
  const normalized = String(scene_name ?? "").trim();
  if (!normalized) {
    throw new Error("Scene name cannot be empty.");
  }
  return normalized.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function parse_content_range(value) {
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(String(value ?? ""));
  if (!match) {
    throw new Error("Scene package chunks require a valid Content-Range header.");
  }
  const start = Number(match[1]);
  const end = Number(match[2]);
  const total = Number(match[3]);
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    !Number.isSafeInteger(total) ||
    start < 0 ||
    end < start ||
    total <= end
  ) {
    throw new Error("Scene package Content-Range is invalid.");
  }
  return { start, end, total };
}

function parse_request_range(value, total_byte_length) {
  if (value === undefined) {
    return {
      start: 0,
      end: total_byte_length - 1,
      partial: false,
    };
  }

  const match = /^bytes=(\d+)-(\d*)$/.exec(String(value));
  if (!match) {
    throw new Error("Scene package request requires a valid byte range.");
  }
  const start = Number(match[1]);
  const end = match[2] ? Number(match[2]) : total_byte_length - 1;
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    end >= total_byte_length
  ) {
    throw new Error("Scene package byte range is outside the saved package.");
  }
  return { start, end, partial: true };
}

async function read_exact(file, buffer, position = 0) {
  let total_read = 0;
  while (total_read < buffer.byteLength) {
    const result = await file.read(
      buffer,
      total_read,
      buffer.byteLength - total_read,
      position + total_read
    );
    if (result.bytesRead === 0) break;
    total_read += result.bytesRead;
  }
  return total_read;
}

async function validate_scene_package_file(file_path, expected_scene_name, expected_byte_length) {
  const file = await fs.open(file_path, "r");
  try {
    const preamble = Buffer.alloc(scene_data_package_fixed_header_byte_length);
    const preamble_bytes_read = await read_exact(file, preamble);
    if (preamble_bytes_read !== preamble.byteLength) {
      throw new Error("Saved scene package does not contain a complete preamble.");
    }
    const header = SceneDataPackage.read_header(preamble);
    if (header.total_byte_length !== expected_byte_length) {
      throw new Error("Saved scene package length does not match its header.");
    }
    if (header.header_byte_length > 64 * 1024 * 1024) {
      throw new Error("Saved scene package directory exceeds the 64 MiB limit.");
    }

    const index = Buffer.alloc(header.header_byte_length);
    const index_bytes_read = await read_exact(file, index);
    if (index_bytes_read !== index.byteLength) {
      throw new Error("Saved scene package does not contain a complete directory.");
    }
    const scene_package = SceneDataPackage.deserialize_index(index, {
      total_byte_length: expected_byte_length,
    });
    if (scene_package.scene_name !== expected_scene_name) {
      throw new Error(
        `Scene package belongs to '${scene_package.scene_name}', not '${expected_scene_name}'.`
      );
    }
  } finally {
    await file.close();
  }
}

async function write_request_range_to_file(request, file_path, range) {
  const expected_byte_length = range.end - range.start + 1;
  const declared_byte_length = Number(request.headers["content-length"]);
  if (Number.isSafeInteger(declared_byte_length) && declared_byte_length !== expected_byte_length) {
    throw new Error("Scene package chunk length does not match Content-Range.");
  }

  if (range.start > 0) {
    const file_stats = await fs.stat(file_path);
    if (file_stats.size === range.end + 1) {
      return false;
    }
    if (file_stats.size !== range.start) {
      throw new Error(
        `Scene package upload expected offset ${file_stats.size}, received ${range.start}.`
      );
    }
  }

  await fs.mkdir(path.dirname(file_path), { recursive: true });
  const file = await fs.open(file_path, range.start === 0 ? "w" : "r+");
  let received_byte_length = 0;
  try {
    for await (const chunk of request) {
      if (received_byte_length + chunk.byteLength > expected_byte_length) {
        throw new Error("Scene package request body exceeds its Content-Range.");
      }

      let chunk_offset = 0;
      while (chunk_offset < chunk.byteLength) {
        const { bytesWritten } = await file.write(
          chunk,
          chunk_offset,
          chunk.byteLength - chunk_offset,
          range.start + received_byte_length + chunk_offset
        );
        if (bytesWritten === 0) {
          throw new Error("Scene package upload stopped before its chunk was written.");
        }
        chunk_offset += bytesWritten;
      }
      received_byte_length += chunk.byteLength;
    }
  } finally {
    await file.close();
  }

  if (received_byte_length !== expected_byte_length) {
    throw new Error(
      `Scene package chunk contained ${received_byte_length} bytes; ${expected_byte_length} were expected.`
    );
  }
  return true;
}

async function save_scene_package_chunk(req, res) {
  let temporary_file_path = null;
  try {
    const project_segments = normalize_project_root(req.query.project_root);
    const scene_name = sanitize_scene_name(req.query.scene_name);
    const range = parse_content_range(req.headers["content-range"]);

    const file_path = path.resolve(
      ASSETS_BASE_PATH,
      ...project_segments,
      "scenes",
      `${scene_name}.scene.bin`
    );
    const relative_path = path.relative(ASSETS_BASE_PATH, file_path);
    if (relative_path.startsWith("..") || path.isAbsolute(relative_path)) {
      return res.status(400).json({ error: "Scene package path is outside assets." });
    }

    temporary_file_path = path.resolve(
      SCENE_UPLOAD_BASE_PATH,
      ...project_segments,
      `${scene_name}.scene.bin.upload.tmp`
    );
    const relative_upload_path = path.relative(SCENE_UPLOAD_BASE_PATH, temporary_file_path);
    if (relative_upload_path.startsWith("..") || path.isAbsolute(relative_upload_path)) {
      return res.status(400).json({ error: "Scene upload path is outside its temporary root." });
    }

    await write_request_range_to_file(req, temporary_file_path, range);

    const received_byte_length = range.end + 1;
    if (received_byte_length < range.total) {
      return res.json({
        success: true,
        complete: false,
        received_byte_length,
      });
    }

    await validate_scene_package_file(
      temporary_file_path,
      String(req.query.scene_name).trim(),
      range.total
    );
    await fs.mkdir(path.dirname(file_path), {
      recursive: true,
    });
    await fs.rename(temporary_file_path, file_path);
    temporary_file_path = null;
    return res.json({
      success: true,
      complete: true,
      asset_path: relative_path.replace(/\\/g, "/"),
      read_url: `/sundown/dev/scene-package?${new URLSearchParams({
        project_root: project_segments.join("/"),
        scene_name: String(req.query.scene_name).trim(),
      })}`,
      byte_length: range.total,
    });
  } catch (save_error) {
    if (temporary_file_path) {
      await fs.unlink(temporary_file_path).catch(() => {});
    }
    if (!res.headersSent && !res.destroyed) {
      return res.status(400).json({ error: save_error.message });
    }
  }
}

async function stream_scene_package(req, res) {
  try {
    const project_segments = normalize_project_root(req.query.project_root);
    const scene_name = sanitize_scene_name(req.query.scene_name);
    const file_path = path.resolve(
      ASSETS_BASE_PATH,
      ...project_segments,
      "scenes",
      `${scene_name}.scene.bin`
    );
    const relative_path = path.relative(ASSETS_BASE_PATH, file_path);
    if (relative_path.startsWith("..") || path.isAbsolute(relative_path)) {
      return res.status(400).json({ error: "Scene package path is outside assets." });
    }

    const file_stats = await fs.stat(file_path);
    const range = parse_request_range(req.headers.range, file_stats.size);
    const response_byte_length = range.end - range.start + 1;
    res.status(range.partial ? 206 : 200);
    res.set({
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
      "Content-Length": String(response_byte_length),
      "Content-Type": "application/octet-stream",
    });
    if (range.partial) {
      res.set("Content-Range", `bytes ${range.start}-${range.end}/${file_stats.size}`);
    }

    const stream = createReadStream(file_path, {
      start: range.start,
      end: range.end,
    });
    stream.on("error", (stream_error) => {
      if (!res.headersSent) {
        res.status(500).json({ error: stream_error.message });
      } else {
        res.destroy(stream_error);
      }
    });
    stream.pipe(res);
  } catch (read_error) {
    const status = read_error.code === "ENOENT" ? 404 : 400;
    if (!res.headersSent && !res.destroyed) {
      return res.status(status).json({ error: read_error.message });
    }
  }
}

async function create_dev_server() {
  const app = express();

  // Register binary uploads before Vite so large request streams never pass
  // through development middleware or accumulate in an Express body buffer.
  app.post("/sundown/dev/save-scene-package", save_scene_package_chunk);
  app.get("/sundown/dev/scene-package", stream_scene_package);

  // Middleware to parse JSON bodies
  app.use(express.json());

  // Create Vite server in middleware mode
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: "custom",
    configFile: path.resolve(__dirname, "../vite.config.mjs"),
  });

  // Use Vite's connect instance as middleware
  app.use(vite.middlewares);

  // Serve index.html
  app.get("/", async (req, res, next) => {
    const url = req.originalUrl;

    try {
      let template = await fs.readFile(path.resolve(__dirname, "../index.html"), "utf-8");

      template = await vite.transformIndexHtml(url, template);

      res.status(200).set({ "Content-Type": "text/html" }).end(template);
    } catch (e) {
      vite.ssrFixStacktrace(e);
      next(e);
    }
  });

  // API endpoint to save config
  app.post("/sundown/dev/save-config", async (req, res) => {
    const { file_name, config } = req.body;
    if (!file_name || !config) {
      return res.status(400).json({ error: "Missing file_name or config in request body." });
    }

    try {
      const file_path = path.join(CONFIG_BASE_PATH, `${file_name}.json`);
      await fs.writeFile(file_path, JSON.stringify(config, null, 2), "utf8");
      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // API endpoint to load config
  app.get("/sundown/dev/get-config", async (req, res) => {
    const { file_name } = req.query;
    if (!file_name) {
      return res.status(400).json({ error: "Missing file_name in query parameters." });
    }

    try {
      const file_path = path.join(CONFIG_BASE_PATH, `${file_name}.json`);
      const data = await fs.readFile(file_path, "utf8");
      const config = JSON.parse(data);
      res.json(config);
    } catch (err) {
      if (err.code === "ENOENT") {
        return res.status(404).json({ error: "Config file not found." });
      }
      res.status(500).json({ error: err.message });
    }
  });

  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(
      "\x1b[32mSundown Vite dev server running at \x1b[37mhttp://localhost:" +
        port +
        "\x1b[32m\x1b[0m"
    );
  });
}

create_dev_server().catch((err) => {
  console.error("Failed to start dev server:", err);
  process.exit(1);
});
