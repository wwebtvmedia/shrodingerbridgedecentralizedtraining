import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        worker: resolve(__dirname, 'src/network/worker.js')
      }
    }
  },
  server: {
    port: 3000,
    open: true
  },
  optimizeDeps: {
    exclude: ['@web/torch']
  }
})