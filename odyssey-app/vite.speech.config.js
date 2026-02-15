import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  root: path.resolve(__dirname, '../experiments/speech-to-text'),
  server: {
    port: 5174,
    open: true
  }
});
