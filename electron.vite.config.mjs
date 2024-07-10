import { defineConfig } from "vite";

const all_config = {
  main: {
    build: {
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
      rollupOptions: {
        input: "./dist/index.html",
      },
    },
  },
  preload: {
    build: {
      rollupOptions: {
        input: {
          index: "./electron/preload/preload.js",
        },
      },
    },
  },
}

export default defineConfig(all_config);
