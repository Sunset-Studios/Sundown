import { defineConfig } from "vite";

const all_config = {
  main: {
    build: {
      outDir: "./dist/electron/main",
      rollupOptions: {
        input: {
          index: "./electron/main/main.js",
        },
      },
    },
  },
  renderer: {
    root: "./dist",
    build: {
      outDir: "./dist/electron/renderer",
      rollupOptions: {
        input: "./dist/index.html",
      },
    },
  },
  preload: {
    build: {
      outDir: "./dist/electron/preload",
      rollupOptions: {
        input: {
          index: "./electron/preload/preload.js",
        },
      },
    },
  },
}

export default defineConfig(all_config);
