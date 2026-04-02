import { create_job_result } from "./job_worker_runtime.js";

function resolve_href(path, base_href = self.location.href) {
  return new URL(path, base_href).href;
}

function is_html_content_type(content_type = "") {
  return content_type.includes("text/html") || content_type.includes("application/xhtml+xml");
}

export async function load_texture_bitmaps_job(payload, { signal, post_progress }) {
  const paths = payload?.paths ?? [];
  const base_url = payload?.base_url ?? self.location.href;
  if (!Array.isArray(paths) || paths.length === 0) {
    throw new Error("Texture load job requires one or more paths");
  }

  const resolved_paths = paths.map((path) => resolve_href(path, base_url));
  const bitmaps = [];
  const transferables = [];

  for (let i = 0; i < resolved_paths.length; i++) {
    const url = resolved_paths[i];
    const response = await fetch(url, { signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch texture '${url}': ${response.status} ${response.statusText}`);
    }

    const content_type = response.headers.get("content-type") || "";
    if (is_html_content_type(content_type)) {
      throw new Error(`Expected image asset at '${url}', but received HTML`);
    }

    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob, {
      colorSpaceConversion: "none",
    });

    bitmaps.push(bitmap);
    transferables.push(bitmap);

    post_progress?.({
      loaded: i + 1,
      total: resolved_paths.length,
      path: paths[i],
      url,
    });
  }

  return create_job_result(
    {
      bitmaps,
      resolved_paths,
    },
    transferables
  );
}
