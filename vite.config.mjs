import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        main: "index.html",
      },
    },
  },
  publicDir: "assets",
  define: {
    __DEV__: "true",
  },
  server: {
    // if you’re proxying through Express you may not need this;
    // otherwise Vite itself must serve these headers.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
