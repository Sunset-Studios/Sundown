import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
    sourcemap: process.env.SUNDOWN_SMOKE_BUILD === "1",
    minify: process.env.SUNDOWN_SMOKE_BUILD === "1" ? false : "esbuild",
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
    watch: {
      ignored: ["**/executables/**"],
    },
    // if you’re proxying through Express you may not need this;
    // otherwise Vite itself must serve these headers.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
