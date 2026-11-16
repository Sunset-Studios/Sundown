import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { sample_scenes, select_sample_scenes } from "../samples/registry.js";
import { SmokeSuiteReport } from "../../tools/smoke/suite_report.js";

test("all registered sample scenes are selected by default in interactive scene order", () => {
  assert.ok(sample_scenes.length > 0);
  assert.equal(new Set(sample_scenes.map((sample) => sample.id)).size, sample_scenes.length);
  assert.deepEqual(
    select_sample_scenes().map((sample) => sample.id),
    sample_scenes.map((sample) => sample.id)
  );
  assert.equal(sample_scenes[0].id, "SolarECSTestScene");
  const sample_directory = new URL("../samples/", import.meta.url);
  const exported_scenes = readdirSync(sample_directory)
    .filter((file) => file.endsWith(".js"))
    .flatMap((file) => {
      const source = readFileSync(new URL(file, sample_directory), "utf8");
      return [...source.matchAll(/export class (\w+) extends Scene/g)].map((match) => match[1]);
    });
  assert.deepEqual(sample_scenes.map((sample) => sample.id).sort(), exported_scenes.sort());
});

test("scene filters accept comma lists, repeated flags, case and snake_case", () => {
  const selected = select_sample_scenes(["UI3DTestScene,bvh_scene", "BVHScene", "gltfmodelscene"]);
  assert.deepEqual(
    selected.map((sample) => sample.id),
    ["BVHScene", "GLTFModelScene", "UI3DTestScene"]
  );
  assert.throws(() => select_sample_scenes(["UnknownScene"]), /Unknown sample/);
  assert.throws(() => select_sample_scenes([""]), /Unknown sample/);
  assert.throws(() => select_sample_scenes(["BVHScene,"]), /Unknown sample/);
});

test("sample registry lists scenes without loading browser-only modules", () => {
  const result = spawnSync(
    process.execPath,
    ["tools/smoke.js", "--catalog", "example/smoke.config.js", "--list"],
    {
      cwd: fileURLToPath(new URL("../../", import.meta.url)),
      encoding: "utf8",
    }
  );
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(
    result.stdout.trim().split("\n"),
    sample_scenes.map((sample) => sample.id)
  );
});

test("every sample catalog entry has a separate exported scene module", () => {
  const registry = readFileSync(new URL("../samples/registry.js", import.meta.url), "utf8");
  const paths = [...registry.matchAll(/import\("(\.\/[^\"]+)"\)/g)].map((match) => match[1]);
  assert.equal(paths.length, sample_scenes.length);
  for (let index = 0; index < paths.length; index++) {
    const source = readFileSync(new URL(`../samples/${paths[index]}`, import.meta.url), "utf8");
    assert.ok(source.includes(`export class ${sample_scenes[index].id} extends Scene`));
    assert.ok(!source.includes("SmokeScene"));
  }
});

test("suite success requires every selected scene and preserves failures", () => {
  const report = new SmokeSuiteReport(select_sample_scenes(["BVHScene,UI3DTestScene"]));
  report.scenes.push({ sample: "BVHScene", status: "passed" });
  assert.equal(report.finish(), 1);
  report.scenes.push({ sample: "UI3DTestScene", status: "failed" });
  assert.equal(report.finish(), 1);
  report.scenes[1].status = "passed";
  assert.equal(report.finish(), 0);
  report.add({ severity: "error", category: "build", message: "build failed" });
  assert.equal(report.finish(), 1);
  assert.equal(report.toJSON().schema_version, 2);
  assert.equal(report.toJSON().scenes.length, 2);
});
