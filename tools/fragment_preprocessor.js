import { FragmentGenerator } from "../engine/src/core/ecs/meta/fragment_generator.js";

import fs from "fs";
import path from "path";
import prettier from 'prettier';
import { fileURLToPath, pathToFileURL } from "url";

// Get the current file's directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


async function writeFormattedFile(path, content) {
    const prettierConfig = await prettier.resolveConfig(process.cwd());
    
    const formattedContent = await prettier.format(content, {
        ...prettierConfig,
        parser: 'babel',
    });

    fs.writeFileSync(path, formattedContent);
}

function search_fragment_files_in_directory(dir, fragment_files) {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const full_path = path.join(dir, file);
    const stats = fs.statSync(full_path);

    if (stats.isDirectory()) {
      search_fragment_files_in_directory(full_path, fragment_files);
    } else if (file === "fragment_definitions.js") {
      fragment_files.push(full_path);
    }
  }
}

export function find_all_fragment_definition_files() {
  const fragment_files = [];
  const root_dir = process.cwd();

  search_fragment_files_in_directory(root_dir, fragment_files);

  return fragment_files;
}

async function generate_fragments_from_definitions(file_path, output_path) {
  const definitions_path = pathToFileURL(path.resolve(__dirname, file_path));
  const { definitions } = await import(definitions_path);
  if (definitions && definitions.length > 0) {
    for (const definition of definitions) {
      const fragment = FragmentGenerator.generate(definition);
      const snake_case_name = definition.name.replace(/([A-Z])/g, '_$1').toLowerCase().substring(1);
      await writeFormattedFile(`${output_path}/${snake_case_name}_fragment.js`, fragment);
    }
  }
}

export async function process_all_fragment_definitions() {
  const fragment_files = find_all_fragment_definition_files();
  for (const file of fragment_files) {
    await generate_fragments_from_definitions(file, path.dirname(file));
  }
}

process_all_fragment_definitions()