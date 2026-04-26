import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/analyze": "http://localhost:8001",
      "/analyze-text": "http://localhost:8001",
      "/monitor": "http://localhost:8001",
      "/gate": "http://localhost:8001",
      "/mitigate": "http://localhost:8001",
      "/mitigate-and-analyze": "http://localhost:8001",
      "/validate": "http://localhost:8001",
      "/auto-debias": "http://localhost:8001",
      "/auto-debias-analyze": "http://localhost:8001",
      "/gate-decision": "http://localhost:8001",
      "/audit-log": "http://localhost:8001",
      "/health": "http://localhost:8001"
    }
  }
});
