import {
  create_job_result,
  resolve_href,
  is_html_content_type,
} from "../../utility/job_worker_runtime.js";

export async function load_shader_source_text_job(payload, { signal, post_progress }) {
  const path = payload?.path;
  const base_url = payload?.base_url ?? self.location.href;

  if (!path) {
    throw new Error("Shader load job requires a shader path");
  }

  const url = resolve_href(path, base_url);
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch shader '${url}': ${response.status} ${response.statusText}`);
  }

  const content_type = response.headers.get("content-type") || "";
  if (is_html_content_type(content_type)) {
    throw new Error(`Expected shader asset at '${url}', but received HTML`);
  }

  const source = await response.text();
  if (source.includes("<!DOCTYPE html>")) {
    throw new Error(`Expected shader asset at '${url}', but received HTML`);
  }

  post_progress?.({
    loaded: 1,
    total: 1,
    path,
    url,
  });

  return create_job_result({
    path,
    url,
    source,
  });
}
