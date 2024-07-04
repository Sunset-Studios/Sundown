import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./engine/src', import.meta.url)),
      '#': fileURLToPath(new URL('./', import.meta.url))
    }
  },
  assetsInclude: ['engine/shaders/**']
})