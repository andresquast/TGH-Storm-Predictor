import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const base = process.env.NODE_ENV === 'production' ? '/TGH-Storm-Predictor/' : '/';

export default defineConfig({
  plugins: [
    react()
  ],
  base: base,
  server: {
    port: 3000,
    proxy: {
      "/predict": {
        target: "http://127.0.0.1:5001", // Use IPv4 explicitly to avoid IPv6 connection issues
        changeOrigin: true,
        secure: false,
        ws: false,
      },
      "/health": {
        target: "http://127.0.0.1:5001", // Use IPv4 explicitly
        changeOrigin: true,
        secure: false,
      },
      "/api": {
        target: "http://127.0.0.1:5001", // Use IPv4 explicitly
        changeOrigin: true,
        secure: false,
        ws: false,
      },
    },
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
  },
});
