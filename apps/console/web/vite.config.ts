import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.CONSOLE_BACKEND ?? 'http://127.0.0.1:8790'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
    },
  },
  build: {
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/echarts') || id.includes('node_modules/zrender')) return 'echarts'
          if (id.includes('node_modules/antd') || id.includes('node_modules/@ant-design') || id.includes('node_modules/rc-')) return 'antd'
          return undefined
        },
      },
    },
  },
})
