import { createServer as createViteServer } from "vite";
import express from "express";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONFIG_BASE_PATH = path.resolve(__dirname, "../assets/config");

async function create_dev_server() {
  const app = express();

  // Middleware to parse JSON bodies
  app.use(express.json());

  // Create Vite server in middleware mode
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: "custom",
    configFile: path.resolve(__dirname, "../vite.config.mjs"),
  });

  app.use((req, res, next) => {
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    next();
  });

  // Use Vite's connect instance as middleware
  app.use(vite.middlewares);

  // Serve general requests
  app.use("*", async (req, res, next) => {
    next();
  });

  // Serve index.html
  app.get("/", async (req, res, next) => {
    const url = req.originalUrl;

    try {
      let template = await fs.readFile(path.resolve(__dirname, "../index.html"), "utf-8");

      template = await vite.transformIndexHtml(url, template);

      res.status(200).set({ "Content-Type": "text/html" }).end(template);
    } catch (e) {
      vite.ssrFixStacktrace(e);
      next(e);
    }
  });

  // API endpoint to save config
  app.post("/sundown/dev/save-config", async (req, res) => {
    const { file_name, config } = req.body;
    if (!file_name || !config) {
      return res.status(400).json({ error: "Missing file_name or config in request body." });
    }

    try {
      const file_path = path.join(CONFIG_BASE_PATH, `${file_name}.json`);
      await fs.writeFile(file_path, JSON.stringify(config, null, 2), "utf8");
      res.json({ success: true });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // API endpoint to load config
  app.get("/sundown/dev/get-config", async (req, res) => {
    const { file_name } = req.query;
    if (!file_name) {
      return res.status(400).json({ error: "Missing file_name in query parameters." });
    }

    try {
      const file_path = path.join(CONFIG_BASE_PATH, `${file_name}.json`);
      const data = await fs.readFile(file_path, "utf8");
      const config = JSON.parse(data);
      res.json(config);
    } catch (err) {
      if (err.code === "ENOENT") {
        return res.status(404).json({ error: "Config file not found." });
      }
      res.status(500).json({ error: err.message });
    }
  });

  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(
      "\x1b[32mSundown Vite dev server running at \x1b[37mhttp://localhost:" +
        port +
        "\x1b[32m\x1b[0m"
    );
  });
}

create_dev_server().catch((err) => {
  console.error("Failed to start dev server:", err);
  process.exit(1);
});
