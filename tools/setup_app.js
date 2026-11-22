// ═══════════════════════════════════════════════════════════════════════════════
// Application bootstrap — generate a starter without replacing application work.
// Templates and smoke integration remain tooling-owned, not engine policies.
// ═══════════════════════════════════════════════════════════════════════════════

import {
  closeSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  rmSync,
  rmdirSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { format, resolveConfig } from "prettier";

function require_absent(filename) {
  try {
    lstatSync(filename);
  } catch (error) {
    if (error.code === "ENOENT") return;
    throw error;
  }
  throw new Error(
    `Refusing to overwrite '${filename}'. Move it aside or integrate the starter manually.`
  );
}

function read_regular_file(filename) {
  if (!lstatSync(filename).isFile()) {
    throw new Error(`Expected a regular file, not a directory or symlink: '${filename}'.`);
  }
  return readFileSync(filename, "utf8");
}

function replace_module_entry(html) {
  const modules = [...html.matchAll(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi)].filter((match) =>
    /\stype\s*=\s*(["'])module\1/i.test(match[0])
  );
  if (modules.length !== 1 || !/\ssrc\s*=\s*(["'])[^"']+\1/i.test(modules[0][0])) {
    throw new Error("index.html must have exactly one module script with a quoted src attribute.");
  }
  for (const id of ["gpu-canvas", "ui-canvas"]) {
    if (!new RegExp(`<canvas\\b[^>]*\\sid\\s*=\\s*(["'])${id}\\1`, "i").test(html)) {
      throw new Error(`index.html is missing the '${id}' canvas required by the starter.`);
    }
  }
  const module = modules[0];
  const replacement = module[0].replace(/(\ssrc\s*=\s*)(["'])[^"']+\2/i, '$1"app/app.js"');
  return html.slice(0, module.index) + replacement + html.slice(module.index + module[0].length);
}

function get_example_directories(root) {
  return ["example", "assets/example"].filter((directory) => {
    const entry = lstatSync(path.join(root, directory), { throwIfNoEntry: false });
    if (!entry) return false;
    if (!entry.isDirectory()) {
      throw new Error(
        `Refusing to remove '${directory}': expected a directory, not a symlink or file.`
      );
    }
    return true;
  });
}

export async function setup_application(
  root,
  { name = "my-app", dry_run = false, remove_example = false } = {}
) {
  if (!/^[a-z][a-z0-9-]*$/.test(name)) {
    throw new Error(
      "--name must start with a lowercase letter and contain only lowercase letters, digits and hyphens."
    );
  }
  root = path.resolve(root);
  const app_directory = path.join(root, "app");
  const asset_directory = path.join(root, "assets", "app");
  require_absent(app_directory);
  require_absent(asset_directory);
  if (!lstatSync(path.join(root, "assets")).isDirectory()) {
    throw new Error("Expected an assets directory, not a symlink or file.");
  }

  const index_path = path.join(root, "index.html");
  const package_path = path.join(root, "package.json");
  const original_index = read_regular_file(index_path);
  const original_package = read_regular_file(package_path);
  const updated_index = replace_module_entry(original_index);
  const package_data = JSON.parse(original_package);
  if (
    package_data.type !== "module" ||
    !package_data.scripts ||
    typeof package_data.scripts !== "object" ||
    Array.isArray(package_data.scripts)
  ) {
    throw new Error(
      "Expected a package.json with type: module and a scripts object. Run from your Sundown repository root."
    );
  }
  if (Object.hasOwn(package_data.scripts, "smoke:app")) {
    throw new Error("Refusing to replace the existing smoke:app npm script.");
  }
  package_data.scripts["smoke:app"] = "node tools/smoke.js --catalog app/smoke.config.js";
  const removed_scripts = remove_example
    ? ["smoke:example", "test:example"].filter((script) =>
        Object.hasOwn(package_data.scripts, script)
      )
    : [];
  for (const script of removed_scripts) delete package_data.scripts[script];

  const prettier_config = (await resolveConfig(package_path)) ?? {};
  const generated_files = new Map();
  for (const filename of ["app.js", "main_scene.js", "smoke.config.js"]) {
    const template = readFileSync(
      new URL(`./templates/application/${filename}`, import.meta.url),
      "utf8"
    );
    const source = template.replaceAll("__APP_NAME__", name);
    generated_files.set(
      path.join(app_directory, filename),
      await format(source, { ...prettier_config, parser: "babel" })
    );
  }
  generated_files.set(path.join(asset_directory, ".gitkeep"), "");
  const updated_package = await format(JSON.stringify(package_data), {
    ...prettier_config,
    parser: "json",
  });
  const removed_directories = remove_example ? get_example_directories(root) : [];
  const changes = {
    written_files: [...generated_files.keys(), index_path, package_path].map((filename) =>
      path.relative(root, filename)
    ),
    removed_directories,
    removed_scripts,
  };
  if (dry_run) return changes;

  // Formatting is asynchronous. Recheck ownership before any filesystem writes.
  require_absent(app_directory);
  require_absent(asset_directory);
  if (
    read_regular_file(index_path) !== original_index ||
    read_regular_file(package_path) !== original_package
  ) {
    throw new Error(
      "index.html or package.json changed during setup. No files were written; rerun after reviewing your changes."
    );
  }

  const created_directories = [];
  const created_files = [];
  const modified_files = new Map();
  try {
    for (const directory of [app_directory, asset_directory]) {
      mkdirSync(directory);
      created_directories.push(directory);
    }
    for (const [filename, content] of generated_files) {
      const descriptor = openSync(filename, "wx");
      created_files.push(filename);
      try {
        writeFileSync(descriptor, content);
      } finally {
        closeSync(descriptor);
      }
    }
    for (const [filename, original, updated] of [
      [index_path, original_index, updated_index],
      [package_path, original_package, updated_package],
    ]) {
      modified_files.set(filename, original);
      writeFileSync(filename, updated);
    }
  } catch (error) {
    try {
      for (const [filename, original] of modified_files) {
        if (read_regular_file(filename) !== original) writeFileSync(filename, original);
      }
      for (const filename of created_files.reverse()) unlinkSync(filename);
      for (const directory of created_directories.reverse()) rmdirSync(directory);
    } catch (rollback_error) {
      throw new AggregateError(
        [error, rollback_error],
        `Setup failed and rollback was incomplete: ${error.message}; ${rollback_error.message}`
      );
    }
    throw error;
  }
  // Don't touch sample files until the new app and its entry point are in place.
  for (const directory of removed_directories) {
    try {
      rmSync(path.join(root, directory), { recursive: true });
    } catch (error) {
      throw new Error(
        `Starter created, but removal of '${directory}/' did not finish. Review remaining sample directories and remove them manually: ${error.message}`,
        { cause: error }
      );
    }
  }
  return changes;
}

async function main() {
  const { values } = parseArgs({
    options: {
      name: { type: "string", default: "my-app" },
      "dry-run": { type: "boolean", default: false },
      "remove-example": { type: "boolean", default: false },
      help: { type: "boolean", default: false },
    },
  });
  if (values.help) {
    console.log(
      "Usage: npm run app:init -- [--name my-app] [--dry-run] [--remove-example]\nCreates app/ and assets/app/, selects app/app.js in index.html, and adds smoke:app.\n--remove-example deletes example/ and assets/example/ (including local files) and removes the sample npm scripts. By default, sample content is kept.\nExisting app files or smoke:app scripts are never overwritten."
    );
    return;
  }
  const changes = await setup_application(process.cwd(), {
    name: values.name,
    dry_run: values["dry-run"],
    remove_example: values["remove-example"],
  });
  console.log(
    values["dry-run"] ? "Planned changes (no files written):" : "Application starter created:"
  );
  for (const filename of changes.written_files) console.log(`  ${filename}`);
  const removal_label = values["dry-run"] ? "Would remove" : "Removed";
  for (const directory of changes.removed_directories)
    console.log(`  ${removal_label} ${directory}/`);
  for (const script of changes.removed_scripts)
    console.log(`  ${removal_label} npm script ${script}`);
  if (!values["dry-run"]) console.log("\nNext: npm run dev\nVerify: npm run smoke:app");
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  void main().catch((error) => {
    console.error(`Application setup failed: ${error.message}`);
    process.exitCode = 1;
  });
}
