import { readFile, writeFile } from "node:fs/promises";
import { format } from "@wasm-fmt/wgslfmt";

const args = process.argv.slice(2);
const check = args.includes("--check");
const paths = args.filter((arg) => arg !== "--check");

if (paths.length === 0 || paths.some((path) => !path.endsWith(".wgsl"))) {
  console.error("Usage: npm run format:wgsl -- [--check] path/to/file.wgsl [...]");
  process.exit(1);
}

// Sundown preprocesses these directives before compiling WGSL. Refuse to write
// if a formatter version ever alters them, rather than silently changing builds.
function directives(source) {
  return source.match(/^\s*#.*$/gm)?.map((line) => line.trim()) ?? [];
}

let failed = false;
for (const path of paths) {
  try {
    const source = await readFile(path, "utf8");
    const formatted = format(source, { indent_symbol: "    ", trailing_commas: "ignore" });
    if (JSON.stringify(directives(source)) !== JSON.stringify(directives(formatted))) {
      throw new Error("formatter changed preprocessing directives");
    }
    if (source !== formatted) {
      if (check) {
        console.error(`${path}: formatting differs`);
        failed = true;
      } else {
        await writeFile(path, formatted);
        console.log(`Formatted ${path}`);
      }
    }
  } catch (error) {
    console.error(`${path}: ${error.message}`);
    failed = true;
  }
}
if (failed) process.exitCode = 1;
