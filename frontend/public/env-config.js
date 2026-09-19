// Overwritten at container startup by docker-entrypoint.sh (envsubst) so the
// same built image can point at different API Gateways without a rebuild.
// Local `npm run dev`/`vite build` never touches this file -- it falls back
// to VITE_API_BASE_URL from .env at build time (see src/api/client.ts).
window.__ENV__ = {
  VITE_API_BASE_URL: "",
};
