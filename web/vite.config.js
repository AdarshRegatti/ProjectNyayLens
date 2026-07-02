import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react' // Fix this line right here!

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
  }
})