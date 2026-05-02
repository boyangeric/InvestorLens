import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      // The backend mounts the chat router at /chat, so the WS endpoint is
      // /chat/ws/{id}. We keep the frontend URL as /ws/{id} and rewrite here
      // so the hook stays oblivious to the backend route layout.
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        rewrite: (path) => `/chat${path}`,
      },
    },
  },
});
