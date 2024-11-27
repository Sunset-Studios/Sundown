import generateBMFont from "msdf-bmfont-xml";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FONTS_BASE = `${__dirname}/../assets`;

function find_font_files(dir, font_files) {
  const font_dir_files = fs.readdirSync(dir);
  for (const font_file of font_dir_files) {
    const full_path = path.join(dir, font_file);
    const stats = fs.statSync(full_path);

    if (stats.isDirectory()) {
      find_font_files(full_path, font_files);
    } else if (font_file.match(/\.(ttf|otf)$/i)) {
      font_files.push(full_path);
    }
  }
}

function find_fonts(dir, font_files) {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const full_path = path.join(dir, file);
    const stats = fs.statSync(full_path);

    if (stats.isDirectory()) {
      if (path.basename(full_path) === "fonts") {
        find_font_files(full_path, font_files);
      }
      find_fonts(full_path, font_files);
    }
  }
}

const font_files = [];
find_fonts(FONTS_BASE, font_files);

for (const font_file of font_files) {
  generateBMFont(font_file, {
    outputType: "json",
  }, (error, textures, font) => {
    if (error) throw error;

    // Write font textures
    textures.forEach((texture, index) => {
      const png_filename = texture.filename.endsWith(".png")
        ? texture.filename
        : texture.filename + ".png";
      fs.writeFile(png_filename, texture.texture, (err) => {
        if (err) throw err;
      });
    });

    // Write font data file
    fs.writeFile(font.filename, font.data, (err) => {
      if (err) throw err;
    });

    // Update font manifest
    const manifest_path = path.join(path.dirname(font.filename), 'font_manifest.json');
    let manifest = { fonts: [] };
    
    // Read existing manifest if it exists
    if (fs.existsSync(manifest_path)) {
      manifest = JSON.parse(fs.readFileSync(manifest_path));
    }

    // Add/update font entry
    const relative_path = path.relative(FONTS_BASE, font.filename).split(path.sep).join('/');
    const font_entry = {
      name: path.basename(relative_path, path.extname(relative_path)),
      path: relative_path
    };
    
    // Check if font already exists in manifest
    const existing_index = manifest.fonts.findIndex(f => f.path === relative_path);
    if (existing_index >= 0) {
      manifest.fonts[existing_index] = font_entry;
    } else {
      manifest.fonts.push(font_entry);
    }

    // Write updated manifest
    fs.writeFileSync(manifest_path, JSON.stringify(manifest, null, 2));
  });
}
