#!/bin/sh
# Runs as part of nginx's standard /docker-entrypoint.d/ hook chain (nginx:alpine
# executes every executable script there before starting), so no ENTRYPOINT
# override is needed -- this just has to be executable and in that directory.
set -eu
cat > /usr/share/nginx/html/env-config.js <<JS
window.__ENV__ = {
  VITE_API_BASE_URL: "${API_BASE_URL}",
};
JS
