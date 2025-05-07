import { FragmentGenerator } from "../engine/src/core/ecs/meta/fragment_generator.js";

import fs from "fs";
import path from "path";
import prettier from "prettier";
import { fileURLToPath, pathToFileURL } from "url";

// Get the current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- Define the location for the single, consolidated registry ---
const CONSOLIDATED_REGISTRY_PATH = path.resolve(
  __dirname,
  "../engine/src/core/ecs/fragment_registry.js"
);

// Base directory for engine imports (used for calculating relative paths for fragment generation)
const ENGINE_BASE_DIR = path.resolve(__dirname, "../engine/src/core/ecs/fragments");

async function writeFormattedFile(filePath, content) {
  try {
    // Ensure the directory exists before writing
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const prettier_config = (await prettier.resolveConfig(process.cwd())) || {}; // Provide default empty config
    const formatted_content = await prettier.format(content, {
      ...prettier_config,
      parser: "babel", // Use babel parser for JS files
    });
    fs.writeFileSync(filePath, formatted_content);
    console.log(`Successfully wrote formatted file: ${filePath}`);
  } catch (error) {
    console.error(`Error writing formatted file ${filePath}:`, error);
    // Fallback to writing unformatted content if formatting fails
    fs.writeFileSync(filePath, content);
  }
}

function search_fragment_files_in_directory(dir, fragment_files) {
  try {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const full_path = path.join(dir, file);
      try {
        const stats = fs.statSync(full_path);
        if (stats.isDirectory()) {
          search_fragment_files_in_directory(full_path, fragment_files);
        } else if (file === "fragment_definitions.js") {
          fragment_files.push(full_path);
        }
      } catch (stat_error) {
        console.warn(`Could not stat file/dir: ${full_path}`, stat_error.code);
      }
    }
  } catch (read_error) {
    console.warn(`Could not read directory: ${dir}`, read_error.code);
  }
}

export function find_all_fragment_definition_files() {
  const fragment_files = [];
  const root_dir = process.cwd(); // Start search from project root

  search_fragment_files_in_directory(root_dir, fragment_files);

  return fragment_files;
}

// This function now returns info needed for the consolidated registry
async function generate_fragments_from_definitions(file_path, output_path) {
  // Relative path from *output* dir to engine base, used *inside* generated fragments
  const relative_path_to_engine = path.relative(output_path, ENGINE_BASE_DIR);

  const definitions_path = pathToFileURL(path.resolve(__dirname, file_path));
  let generated_fragments_info = []; // To store { class_name, absolute_file_path }

  try {
    const { definitions } = await import(definitions_path);
    if (definitions && definitions.length > 0) {
      for (const definition of definitions) {
        // Generate fragment content
        const fragment_content = FragmentGenerator.generate(definition, relative_path_to_engine);

        // Determine output file name and path
        const snake_case_name = definition.name
          .replace(/([A-Z])/g, "_$1")
          .toLowerCase()
          .substring(1);
        const output_file_name = `${snake_case_name}_fragment.js`;
        const absolute_output_file_path = path.resolve(output_path, output_file_name); // Get absolute path

        // Write the generated fragment file
        await writeFormattedFile(absolute_output_file_path, fragment_content);

        // Store info for the registry using ABSOLUTE path
        generated_fragments_info.push({
          class_name: definition.name + "Fragment",
          absolute_file_path: absolute_output_file_path,
        });
      }
    }
  } catch (error) {
    console.error(`Error processing definitions from ${file_path}:`, error);
  }
  return generated_fragments_info;
}

// Generates the SINGLE consolidated registry file
async function generate_consolidated_registry(registry_file_path, all_fragments_info) {
  if (all_fragments_info.length === 0) {
    console.log(`No fragments found, skipping consolidated registry generation.`);
    // Optionally write an empty registry file
    const empty_content = `// Auto-generated: No fragments found.\nexport const ALL_FRAGMENT_CLASSES = [];\n`;
    await writeFormattedFile(registry_file_path, empty_content);
    return;
  }

  const registry_dir = path.dirname(registry_file_path);

  // Sort fragments alphabetically by class name for consistent ordering
  all_fragments_info.sort((a, b) => a.class_name.localeCompare(b.class_name));

  const import_statements = all_fragments_info
    .map((info) => {
      // Calculate path relative from the registry file to the fragment file
      let relative_path = path.relative(registry_dir, info.absolute_file_path);
      // Ensure relative paths start with './' or '../' and use forward slashes
      relative_path = relative_path.replace(/\\/g, "/");
      if (!relative_path.startsWith(".")) {
        relative_path = "./" + relative_path;
      }
      return `import { ${info.class_name} } from "${relative_path}";`;
    })
    .join("\n");

  const export_array = `export const ALL_FRAGMENT_CLASSES = [\n    ${all_fragments_info.map((info) => info.class_name).join(",\n    ")}\n];`;

  const registry_content = `// Auto-generated by fragment_preprocessor.js
// Do not edit this file directly.

${import_statements}

/**
 * A consolidated collection of all known Fragment classes in the project
 * that might require GPU buffer initialization.
 */
${export_array}
`;

  await writeFormattedFile(registry_file_path, registry_content);
}

export async function process_all_fragment_definitions() {
  const fragment_definition_files = find_all_fragment_definition_files();
  const all_fragments_info = []; // Single list for all fragments { class_name, absolute_file_path }

  // Generate all fragment files first and collect info into the single list
  for (const file of fragment_definition_files) {
    const output_dir = path.dirname(file); // Fragments are generated alongside definitions
    const generated_info = await generate_fragments_from_definitions(file, output_dir);
    all_fragments_info.push(...generated_info); // Add generated info to the main list
  }

  // Now generate the single registry file using the consolidated info
  await generate_consolidated_registry(CONSOLIDATED_REGISTRY_PATH, all_fragments_info);

  console.log(
    "Fragment preprocessing complete. Consolidated registry generated at:",
    CONSOLIDATED_REGISTRY_PATH
  );
}

// Run the processor
process_all_fragment_definitions().catch((error) => {
  console.error("Fragment preprocessing failed:", error);
  process.exit(1);
});
