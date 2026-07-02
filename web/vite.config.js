import { defineConfig } from 'vite'
import react from '@vitejs/react-react' // or whatever your framework plugin is named

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
  }
})