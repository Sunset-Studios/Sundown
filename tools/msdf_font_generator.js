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
    textures.forEach((texture, index) => {
      const png_filename = texture.filename.endsWith(".png")
        ? texture.filename
        : texture.filename + ".png";
      fs.writeFile(png_filename, texture.texture, (err) => {
        if (err) throw err;
      });
    });
    fs.writeFile(font.filename, font.data, (err) => {
      if (err) throw err;
    });
  });
}
