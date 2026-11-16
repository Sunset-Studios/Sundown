import assert from "node:assert/strict";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { load_scene_catalog, select_scenes } from "../smoke/catalog.js";
import { SmokeSuiteReport } from "../smoke/suite_report.js";

const catalog_path = fileURLToPath(new URL("./fixtures/smoke_catalog.js", import.meta.url));

test("application catalogs preserve order and filter repeated, comma-separated IDs", async () => {
  const scenes = await load_scene_catalog(catalog_path);
  assert.deepEqual(select_scenes(scenes), scenes);
  assert.deepEqual(select_scenes(scenes, ["warning_scene,fixture-scene", "FixtureScene"]), scenes);
  assert.deepEqual(select_scenes(scenes, ["warning_scene"]), [scenes[1]]);
  assert.throws(() => select_scenes(scenes, ["UnknownScene"]), /Unknown scene/);
  assert.throws(() => select_scenes(scenes, [""]), /Unknown scene/);
  assert.throws(() => select_scenes(scenes, ["FixtureScene,"]), /Unknown scene/);
});

test("explicit scene IDs need no catalog and missing configuration fails clearly", async () => {
  assert.deepEqual(await load_scene_catalog(null, ["CustomScene,OtherScene", "CustomScene"]), [
    { id: "CustomScene" },
    { id: "OtherScene" },
  ]);
  await assert.rejects(load_scene_catalog(), /Provide --catalog/);
  await assert.rejects(load_scene_catalog(null, [""]), /nonempty/);
});

test("invalid and ambiguous application catalogs are rejected", async (context) => {
  const directory = mkdtempSync(path.join(tmpdir(), "sundown-catalog-test-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  const invalid_catalogs = [
    [],
    [{ id: "" }],
    [{ id: " padded " }],
    [{ id: "a,b" }],
    [{ id: "A" }, { id: "a" }],
    [{ id: "A-Scene" }, { id: "a_scene" }],
  ];
  for (let index = 0; index < invalid_catalogs.length; index++) {
    const filename = path.join(directory, `invalid_${index}.mjs`);
    writeFileSync(filename, `export default ${JSON.stringify(invalid_catalogs[index])};`);
    await assert.rejects(load_scene_catalog(filename), /catalog|IDs|Duplicate/);
  }
});

test("launcher lists a custom catalog in a fork with no sample application", (context) => {
  const directory = mkdtempSync(path.join(tmpdir(), "sundown-fork-test-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  const root = fileURLToPath(new URL("../../", import.meta.url));
  for (const filename of [
    "tools/smoke.js",
    "tools/smoke",
    "engine/src/utility/diagnostics_report.js",
  ]) {
    mkdirSync(path.dirname(path.join(directory, filename)), { recursive: true });
    cpSync(path.join(root, filename), path.join(directory, filename), { recursive: true });
  }
  writeFileSync(path.join(directory, "package.json"), JSON.stringify({ type: "module" }));
  symlinkSync(path.join(root, "node_modules"), path.join(directory, "node_modules"), "junction");
  cpSync(catalog_path, path.join(directory, "smoke.config.js"));
  assert.equal(existsSync(path.join(directory, "example")), false);
  const result = spawnSync(
    process.execPath,
    ["tools/smoke.js", "--catalog", "smoke.config.js", "--list"],
    { cwd: directory, encoding: "utf8" }
  );
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(result.stdout.trim().split("\n"), ["FixtureScene", "WarningScene"]);

  const explicit = spawnSync(
    process.execPath,
    ["tools/smoke.js", "--scene", "CustomScene", "--list"],
    {
      cwd: directory,
      encoding: "utf8",
    }
  );
  assert.equal(explicit.status, 0, explicit.stderr);
  assert.equal(explicit.stdout.trim(), "CustomScene");

  const unconfigured = spawnSync(process.execPath, ["tools/smoke.js", "--list"], {
    cwd: directory,
    encoding: "utf8",
  });
  assert.equal(unconfigured.status, 1);
  assert.match(unconfigured.stderr, /Provide --catalog/);
});

test("suite success requires every configured application scene and preserves failures", () => {
  const report = new SmokeSuiteReport([{ id: "FixtureScene" }, { id: "WarningScene" }]);
  report.scenes.push({ sample: "FixtureScene", status: "passed" });
  assert.equal(report.finish(), 1);
  report.scenes.push({ sample: "WarningScene", status: "failed" });
  assert.equal(report.finish(), 1);
  report.scenes[1].status = "passed";
  assert.equal(report.finish(), 0);
  report.add({ severity: "error", category: "build", message: "build failed" });
  assert.equal(report.finish(), 1);
});
