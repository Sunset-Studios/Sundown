import path from "node:path";
import { stat } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export function resolve_asset_path(asset_root, request_url) {
  try {
    const url = new URL(request_url);
    if (url.protocol !== "sundown:" || url.host !== "app") return null;
    const pathname = decodeURIComponent(url.pathname);
    if (pathname.includes("\\") || pathname.includes("\0")) return null;
    const asset_path = path.resolve(asset_root, `.${pathname === "/" ? "/index.html" : pathname}`);
    const relative_path = path.relative(asset_root, asset_path);
    if (relative_path.startsWith("..") || path.isAbsolute(relative_path)) return null;
    return asset_path;
  } catch {
    return null;
  }
}

export function parse_byte_range(range_header, byte_length) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(range_header);
  if (!match || (!match[1] && !match[2])) return null;
  const start = match[1] ? Number(match[1]) : Math.max(0, byte_length - Number(match[2]));
  const end = match[1] && match[2] ? Math.min(Number(match[2]), byte_length - 1) : byte_length - 1;
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start > end ||
    start >= byte_length
  ) {
    return null;
  }
  return { start, end };
}

export async function serve_asset(request, asset_root, fetch_file) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("Method not allowed", { status: 405 });
  }
  const asset_path = resolve_asset_path(asset_root, request.url);
  if (!asset_path) return new Response("Invalid asset path", { status: 403 });

  let file_stats;
  try {
    file_stats = await stat(asset_path);
  } catch (read_error) {
    if (read_error.code === "ENOENT" || read_error.code === "ENOTDIR") {
      return new Response("Asset not found", { status: 404 });
    }
    throw read_error;
  }
  if (!file_stats.isFile()) return new Response("Asset not found", { status: 404 });

  const range_header = request.headers.get("Range");
  const byte_range = range_header ? parse_byte_range(range_header, file_stats.size) : null;
  if (range_header && !byte_range) {
    return new Response(null, {
      status: 416,
      headers: { "Content-Range": `bytes */${file_stats.size}` },
    });
  }

  const request_headers = new Headers();
  if (byte_range) request_headers.set("Range", `bytes=${byte_range.start}-${byte_range.end}`);
  const response = await fetch_file(pathToFileURL(asset_path).href, {
    method: request.method,
    headers: request_headers,
  });
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Accept-Ranges", "bytes");
  // Chromium file fetch honors Range but does not supply HTTP partial-response metadata.
  headers.set(
    "Content-Length",
    String(byte_range ? byte_range.end - byte_range.start + 1 : file_stats.size)
  );
  if (byte_range) {
    headers.set("Content-Range", `bytes ${byte_range.start}-${byte_range.end}/${file_stats.size}`);
  }
  return new Response(request.method === "HEAD" ? null : response.body, {
    status: byte_range ? 206 : response.status,
    headers,
  });
}
