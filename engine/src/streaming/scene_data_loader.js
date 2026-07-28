import { read_file_range_async } from "./streaming_io.js";
import {
  SceneDataPackage,
  SceneDataPackageBuilder,
  scene_data_package_fixed_header_byte_length,
} from "./scene_data_package.js";

const scene_package_extension = ".scene.bin";
const scene_package_directory = "scenes";
const scene_package_save_endpoint = "/sundown/dev/save-scene-package";
const scene_package_upload_chunk_byte_length = 2 * 1024 * 1024;
const scene_package_upload_attempt_count = 3;

function normalize_project_root(project_root) {
  const normalized = String(project_root ?? "")
    .replace(/\\/g, "/")
    .replace(/^\/+|\/+$/g, "");
  if (!normalized) {
    return "";
  }

  const segments = normalized.split("/");
  if (
    segments.some(
      (segment) =>
        !segment || segment === "." || segment === ".." || !/^[a-zA-Z0-9._-]+$/.test(segment)
    )
  ) {
    throw new Error(`Project asset root '${project_root}' contains an unsafe path segment.`);
  }
  return segments.join("/");
}

export function sanitize_scene_data_name(scene_name) {
  const normalized = String(scene_name ?? "").trim();
  if (!normalized) {
    throw new Error("Scene name cannot be empty.");
  }
  return normalized.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

export function build_project_scene_data_asset_path(scene_name, project = { root: "" }) {
  const project_root = normalize_project_root(project?.root);
  const scene_file_name = `${sanitize_scene_data_name(scene_name)}${scene_package_extension}`;
  return [project_root, scene_package_directory, scene_file_name].filter(Boolean).join("/");
}

function create_disk_backed_scene_package(asset_path, index, header, options = {}) {
  const revision = options.cache_bust;
  const source_path =
    revision === undefined
      ? asset_path
      : `${asset_path}${asset_path.includes("?") ? "&" : "?"}scene_data_revision=${encodeURIComponent(revision)}`;
  const fetch_options = {
    cache: "no-store",
    ...(options.fetch_options ?? {}),
  };
  return SceneDataPackage.deserialize_index(index, {
    asset_path: source_path,
    total_byte_length: header.total_byte_length,
    read_entry: async (offset, byte_length) => {
      const payload = await read_file_range_async(source_path, offset, offset + byte_length - 1, {
        fetch_options,
      });
      if (payload === null) {
        throw new Error(
          `Scene package range '${offset}-${offset + byte_length - 1}' could not be loaded from '${asset_path}'.`
        );
      }
      return payload;
    },
  });
}

export async function read_scene_data_package_index_async(asset_path, options = {}) {
  const revision = options.cache_bust;
  const source_path =
    revision === undefined
      ? asset_path
      : `${asset_path}${asset_path.includes("?") ? "&" : "?"}scene_data_revision=${encodeURIComponent(revision)}`;
  const fetch_options = {
    cache: "no-store",
    ...(options.fetch_options ?? {}),
  };
  const preamble = await read_file_range_async(
    source_path,
    0,
    scene_data_package_fixed_header_byte_length - 1,
    { fetch_options }
  );
  if (preamble === null) {
    return null;
  }

  const header = SceneDataPackage.read_header(preamble);
  const index =
    header.header_byte_length === preamble.byteLength
      ? preamble
      : await read_file_range_async(source_path, 0, header.header_byte_length - 1, {
        fetch_options,
      });
  if (index === null) {
    return null;
  }

  return create_disk_backed_scene_package(asset_path, index, header, {
    ...options,
    cache_bust: revision,
  });
}

async function write_scene_package_to_dev_server({
  serialization,
  scene_name,
  project,
  on_progress,
}) {
  const query = new URLSearchParams({
    project_root: normalize_project_root(project?.root),
    scene_name,
  });
  let result = null;
  on_progress?.(0);
  for (const chunk of serialization.chunks(scene_package_upload_chunk_byte_length)) {
    const offset = chunk.offset;
    const end = offset + chunk.bytes.byteLength;
    // Give fetch a small, isolated backing buffer. Passing a subarray whose
    // backing store contains the full scene package can keep that entire store
    // pinned or copied by browser networking implementations.
    const request_body = chunk.bytes.slice();
    let response;
    let upload_error = null;
    for (let attempt = 1; attempt <= scene_package_upload_attempt_count; attempt++) {
      try {
        response = await fetch(`${scene_package_save_endpoint}?${query}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-Range": `bytes ${offset}-${end - 1}/${serialization.byte_length}`,
          },
          body: request_body,
        });
        break;
      } catch (attempt_error) {
        upload_error = attempt_error;
        if (attempt < scene_package_upload_attempt_count) {
          await new Promise((resolve) => setTimeout(resolve, attempt * 250));
        }
      }
    }
    if (!response) {
      throw new Error(
        `Scene package upload failed at byte ${offset}: ${upload_error?.message ?? upload_error}`
      );
    }
    if (!response.ok) {
      const response_text = await response.text();
      throw new Error(
        `Scene package save failed at byte ${offset} (${response.status}): ${response_text || response.statusText
        }`
      );
    }
    result = await response.json();
    const expected_complete = end === serialization.byte_length;
    if (Boolean(result?.complete) !== expected_complete) {
      throw new Error(`Scene package save returned an invalid completion state at byte ${end}.`);
    }
    if (!expected_complete && Number(result?.received_byte_length) !== end) {
      throw new Error(
        `Scene package save acknowledged byte ${result?.received_byte_length ?? "unknown"}; ${end} was expected.`
      );
    }
    on_progress?.(end / serialization.byte_length);
  }
  return result;
}

function validate_handler(handler) {
  if (!handler || typeof handler !== "object") {
    throw new Error("Scene data handlers must be objects.");
  }
  const namespace = String(handler.namespace ?? "").trim();
  if (!namespace) {
    throw new Error("Scene data handlers require a namespace.");
  }
  if (handler.load !== undefined && typeof handler.load !== "function") {
    throw new Error(`Scene data handler '${namespace}' has an invalid load hook.`);
  }
  if (handler.save !== undefined && typeof handler.save !== "function") {
    throw new Error(`Scene data handler '${namespace}' has an invalid save hook.`);
  }
  if (handler.unload !== undefined && typeof handler.unload !== "function") {
    throw new Error(`Scene data handler '${namespace}' has an invalid unload hook.`);
  }
  if (handler.after_save !== undefined && typeof handler.after_save !== "function") {
    throw new Error(`Scene data handler '${namespace}' has an invalid after_save hook.`);
  }
  return namespace;
}

/**
 * Owns one scene's durable data package and routes namespaced sections to
 * implementations without exposing asset paths or container offsets to them.
 */
export class SceneDataLoader {
  constructor(scene_name, options = {}) {
    this.scene_name = String(scene_name ?? "").trim();
    if (!this.scene_name) {
      throw new Error("SceneDataLoader requires a scene name.");
    }

    this.project = options.project ?? {
      name: "default",
      root: "",
    };
    this.asset_path = build_project_scene_data_asset_path(this.scene_name, this.project);
    this.read_package = options.read_package ?? null;
    this.open_package = options.open_package ?? read_scene_data_package_index_async;
    this.uses_disk_backed_packages =
      options.read_package === undefined && options.open_package === undefined;
    this.uses_streamed_writes = options.write_package === undefined;
    this.write_package = options.write_package ?? write_scene_package_to_dev_server;
    this.handlers = new Map();
    this.scene_package = SceneDataPackage.empty(this.scene_name);
    this.load_complete = false;
    this.disposed = false;
    this.generation = 0;
  }

  get_context(scene_package = this.scene_package) {
    return {
      scene_name: this.scene_name,
      asset_path: this.asset_path,
      project: this.project,
      scene_package,
      loader: this,
    };
  }

  register_handler(handler) {
    const namespace = validate_handler(handler);
    if (this.handlers.has(namespace)) {
      throw new Error(`Scene data handler '${namespace}' is already registered.`);
    }
    this.handlers.set(namespace, handler);

    if (this.load_complete && !this.disposed) {
      void this._dispatch_handler(handler, this.scene_package);
    }
    return () => this.unregister_handler(namespace);
  }

  connect_handlers(handlers = []) {
    for (const handler of handlers) {
      this.register_handler(handler);
    }
    return this;
  }

  unregister_handler(namespace) {
    const handler = this.handlers.get(namespace);
    if (!handler) {
      return false;
    }
    this.handlers.delete(namespace);
    if (this.load_complete) {
      void handler.unload?.(this.get_context());
    }
    return true;
  }

  async load(options = {}) {
    if (this.disposed) {
      throw new Error(`Cannot load disposed scene data for '${this.scene_name}'.`);
    }

    const generation = ++this.generation;
    const optional = options.optional !== false;
    let scene_package;
    if (this.read_package) {
      const payload = await this.read_package(this.asset_path, {
        optional,
        label: `Scene data for '${this.scene_name}'`,
      });
      scene_package =
        payload === null
          ? null
          : payload instanceof SceneDataPackage
            ? payload
            : SceneDataPackage.deserialize(payload);
    } else {
      scene_package = await this.open_package(this.asset_path, {
        optional,
      });
    }
    if (this.disposed || generation !== this.generation) {
      return null;
    }
    if (scene_package === null && !optional) {
      throw new Error(`Scene package '${this.asset_path}' could not be loaded.`);
    }

    scene_package ??= SceneDataPackage.empty(this.scene_name);
    if (scene_package.scene_name !== this.scene_name) {
      throw new Error(
        `Scene package '${this.asset_path}' belongs to '${scene_package.scene_name
        }', not '${this.scene_name}'.`
      );
    }

    this.scene_package = scene_package;
    this.load_complete = true;
    await Promise.all(
      Array.from(this.handlers.values(), (handler) =>
        this._dispatch_handler(handler, scene_package)
      )
    );
    return scene_package;
  }

  async _dispatch_handler(handler, scene_package) {
    const section = scene_package.get_section(handler.namespace);
    const context = this.get_context(scene_package);
    if (section) {
      await handler.load?.(section, context);
    } else {
      await handler.unload?.(context);
    }
  }

  async serialize() {
    if (this.disposed) {
      throw new Error(`Cannot serialize disposed scene data for '${this.scene_name}'.`);
    }

    const handler_sections = new Map();
    for (const handler of this.handlers.values()) {
      if (!handler.save) {
        continue;
      }
      const section = await handler.save(this.get_context(this.scene_package));
      handler_sections.set(handler.namespace, section);
    }

    const replaced_sections = Array.from(handler_sections, ([namespace, section]) =>
      section === undefined || section === null ? null : namespace
    ).filter(Boolean);
    const builder = await SceneDataPackageBuilder.from_package_async(this.scene_package, {
      exclude_sections: replaced_sections,
    });
    for (const [namespace, section] of handler_sections) {
      if (section === undefined || section === null) continue;
      if (section.remove === true) {
        builder.remove_section(namespace);
      } else {
        builder.set_section(namespace, section);
      }
    }

    return { plan: builder.create_serialization_plan() };
  }

  async save_to_disk(options = {}) {
    const serialized = await this.serialize();
    const byte_length = serialized.plan.byte_length;
    let saved_package_index = null;
    let saved_package_header = null;
    if (this.uses_disk_backed_packages) {
      saved_package_header = SceneDataPackage.read_header(serialized.plan.header_bytes);
      saved_package_index = serialized.plan.header_bytes.slice();
    }
    const materialized_bytes = this.uses_streamed_writes ? null : serialized.plan.materialize();
    const result = await this.write_package({
      serialization: serialized.plan,
      bytes: materialized_bytes,
      scene_name: this.scene_name,
      project: this.project,
      asset_path: this.asset_path,
      on_progress: options.on_progress,
    });
    const saved_package = this.uses_disk_backed_packages
      ? create_disk_backed_scene_package(
        result?.read_url ?? this.asset_path,
        saved_package_index,
        saved_package_header,
        { cache_bust: Date.now() }
      )
      : SceneDataPackage.deserialize(materialized_bytes);
    serialized.plan = null;

    for (const handler of this.handlers.values()) {
      await handler.after_save?.(result, this.get_context(this.scene_package));
    }

    this.scene_package = saved_package;
    this.load_complete = true;
    await Promise.all(
      Array.from(this.handlers.values(), (handler) =>
        this._dispatch_handler(handler, saved_package)
      )
    );
    return {
      ...result,
      asset_path: result?.asset_path ?? this.asset_path,
      byte_length,
    };
  }

  dispose() {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.generation++;
    const context = this.get_context();
    for (const handler of this.handlers.values()) {
      void handler.unload?.(context);
    }
    this.handlers.clear();
  }
}
