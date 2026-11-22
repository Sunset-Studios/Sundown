import assert from "node:assert/strict";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";
import { check } from "prettier";
import { setup_application } from "../setup_app.js";

const script_path = fileURLToPath(new URL("../setup_app.js", import.meta.url));

function create_repository(context) {
  const root = mkdtempSync(path.join(tmpdir(), "sundown-setup-test-"));
  context.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(path.join(root, "assets"));
  const html = `<title>Keep my title</title>
<link rel="stylesheet" href="styles.css">
<canvas id="gpu-canvas"></canvas><canvas id="ui-canvas"></canvas>
<script src='previous/main.js' type='module'></script>\n`;
  const package_data = {
    name: "existing-package",
    type: "module",
    scripts: { dev: "custom-dev", smoke: "node tools/smoke.js" },
    build: { appId: "org.existing.app" },
  };
  writeFileSync(path.join(root, "index.html"), html);
  writeFileSync(path.join(root, "package.json"), JSON.stringify(package_data));
  return { root, html, package_data };
}

test("setup generates a formatted independent application and preserves unrelated configuration", async (context) => {
  const { root, html, package_data } = create_repository(context);
  await setup_application(root, { name: "custom-app" });
  assert.equal(
    readFileSync(path.join(root, "index.html"), "utf8"),
    html.replace("src='previous/main.js'", 'src="app/app.js"')
  );
  const updated_package = JSON.parse(readFileSync(path.join(root, "package.json")));
  assert.equal(
    updated_package.scripts["smoke:app"],
    "node tools/smoke.js --catalog app/smoke.config.js"
  );
  delete updated_package.scripts["smoke:app"];
  assert.deepEqual(updated_package, package_data);
  assert.equal(existsSync(path.join(root, "assets/app/.gitkeep")), true);
  assert.equal(existsSync(path.join(root, "example")), false);

  for (const filename of ["app.js", "main_scene.js", "smoke.config.js"]) {
    const source = readFileSync(path.join(root, "app", filename), "utf8");
    assert.equal(await check(source, { parser: "babel" }), true);
    assert.doesNotMatch(source, /example\/|__APP_NAME__/);
    const result = spawnSync(process.execPath, ["--input-type=module", "--check"], {
      input: source,
      encoding: "utf8",
    });
    assert.equal(result.status, 0, result.stderr);
  }
  assert.match(readFileSync(path.join(root, "app/app.js"), "utf8"), /name: "custom-app"/);
  const catalog = await import(pathToFileURL(path.join(root, "app/smoke.config.js")).href);
  assert.deepEqual(catalog.default, [{ id: "MainScene" }]);
});

test("CLI dry-run previews changes without modifying the repository", (context) => {
  const { root, html, package_data } = create_repository(context);
  const result = spawnSync(process.execPath, [script_path, "--dry-run", "--name", "preview-app"], {
    cwd: root,
    encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /no files written/);
  assert.match(result.stdout, /app[\\/]main_scene.js/);
  assert.equal(readFileSync(path.join(root, "index.html"), "utf8"), html);
  assert.deepEqual(JSON.parse(readFileSync(path.join(root, "package.json"))), package_data);
  assert.equal(existsSync(path.join(root, "app")), false);
  assert.deepEqual(readdirSync(path.join(root, "assets")), []);
});

test("CLI creates the default starter and refuses a repeated setup", (context) => {
  const { root } = create_repository(context);
  mkdirSync(path.join(root, "example"));
  const first = spawnSync(process.execPath, [script_path], { cwd: root, encoding: "utf8" });
  assert.equal(first.status, 0, first.stderr);
  assert.equal(existsSync(path.join(root, "example")), true);
  const source = readFileSync(path.join(root, "app/app.js"), "utf8");
  assert.match(source, /name: "my-app"/);
  const second = spawnSync(process.execPath, [script_path], { cwd: root, encoding: "utf8" });
  assert.equal(second.status, 1);
  assert.match(second.stderr, /Refusing to overwrite/);
  assert.equal(readFileSync(path.join(root, "app/app.js"), "utf8"), source);
});

test("CLI previews and removes only sample content when explicitly requested", (context) => {
  const { root, html, package_data } = create_repository(context);
  package_data.scripts["smoke:example"] = "sample-smoke";
  package_data.scripts["test:example"] = "sample-tests";
  const package_path = path.join(root, "package.json");
  writeFileSync(package_path, JSON.stringify(package_data));
  for (const directory of ["example", "assets/example", "assets/engine"]) {
    mkdirSync(path.join(root, directory));
    writeFileSync(path.join(root, directory, "keep.txt"), directory);
  }
  const run = (...args) =>
    spawnSync(process.execPath, [script_path, "--remove-example", ...args], {
      cwd: root,
      encoding: "utf8",
    });
  const preview = run("--dry-run");
  assert.equal(preview.status, 0, preview.stderr);
  for (const target of [
    "example/",
    "assets/example/",
    "npm script smoke:example",
    "npm script test:example",
  ]) {
    assert.ok(preview.stdout.includes(`Would remove ${target}\n`));
  }
  assert.equal(readFileSync(path.join(root, "index.html"), "utf8"), html);
  assert.deepEqual(JSON.parse(readFileSync(package_path)), package_data);
  for (const directory of ["example", "assets/example"]) {
    assert.equal(readFileSync(path.join(root, directory, "keep.txt"), "utf8"), directory);
  }

  const result = run();
  assert.equal(result.status, 0, result.stderr);
  for (const directory of ["example", "assets/example"]) {
    assert.equal(existsSync(path.join(root, directory)), false);
  }
  assert.equal(existsSync(path.join(root, "app/app.js")), true);
  assert.equal(readFileSync(path.join(root, "assets/engine/keep.txt"), "utf8"), "assets/engine");
  assert.deepEqual(JSON.parse(readFileSync(package_path)).scripts, {
    dev: "custom-dev",
    smoke: "node tools/smoke.js",
    "smoke:app": "node tools/smoke.js --catalog app/smoke.config.js",
  });
});

test("sample removal refuses a symlink in place of a sample directory", async (context) => {
  const { root } = create_repository(context);
  mkdirSync(path.join(root, "example"));
  symlinkSync(path.join(root, "example"), path.join(root, "assets/example"), "junction");
  await assert.rejects(setup_application(root, { remove_example: true }), /Refusing to remove/);
  assert.equal(existsSync(path.join(root, "example")), true);
  assert.equal(existsSync(path.join(root, "app")), false);
});

for (const conflict of [
  "app",
  "assets/app",
  "script",
  "html",
  "canvas",
  "name",
  "package",
  "symlink",
]) {
  test(`setup refuses ${conflict} conflicts before writing`, async (context) => {
    const { root } = create_repository(context);
    const options = {};
    if (conflict === "app" || conflict === "assets/app") mkdirSync(path.join(root, conflict));
    if (conflict === "script") {
      const filename = path.join(root, "package.json");
      const data = JSON.parse(readFileSync(filename));
      data.scripts["smoke:app"] = "custom-command";
      writeFileSync(filename, JSON.stringify(data));
    }
    if (conflict === "html")
      writeFileSync(
        path.join(root, "index.html"),
        "<script type='module'>import './app.js';</script>"
      );
    if (conflict === "canvas")
      writeFileSync(path.join(root, "index.html"), "<script type='module' src='old.js'></script>");
    if (conflict === "name") options.name = "../../outside";
    if (conflict === "package") writeFileSync(path.join(root, "package.json"), "{invalid");
    if (conflict === "symlink")
      symlinkSync(path.join(root, "nonexistent"), path.join(root, "app"), "junction");
    const html = readFileSync(path.join(root, "index.html"), "utf8");
    const package_source = readFileSync(path.join(root, "package.json"), "utf8");
    await assert.rejects(setup_application(root, options));
    assert.equal(readFileSync(path.join(root, "index.html"), "utf8"), html);
    assert.equal(readFileSync(path.join(root, "package.json"), "utf8"), package_source);
    if (conflict !== "app" && conflict !== "symlink")
      assert.equal(existsSync(path.join(root, "app")), false);
  });
}

test(
  "setup rolls back newly generated files if updating the HTML fails",
  { skip: process.platform === "win32" || process.getuid?.() === 0 },
  async (context) => {
    const { root, html, package_data } = create_repository(context);
    mkdirSync(path.join(root, "example"));
    chmodSync(path.join(root, "index.html"), 0o444);
    try {
      await assert.rejects(setup_application(root, { remove_example: true }), /EACCES|EPERM/);
    } finally {
      chmodSync(path.join(root, "index.html"), 0o644);
    }
    assert.equal(existsSync(path.join(root, "app")), false);
    assert.equal(existsSync(path.join(root, "assets/app")), false);
    assert.equal(existsSync(path.join(root, "example")), true);
    assert.equal(readFileSync(path.join(root, "index.html"), "utf8"), html);
    assert.deepEqual(JSON.parse(readFileSync(path.join(root, "package.json"))), package_data);
  }
);
