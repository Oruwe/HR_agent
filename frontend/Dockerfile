# =============================================================================
# Candidate App + Admin Dashboard -- static build served by nginx.
#
# The API Gateway URL is injected at *container start*, not baked in at build
# time: docker-entrypoint.sh substitutes API_BASE_URL into env-config.js
# before nginx starts, so the same image works against any backend without
# rebuilding.
# =============================================================================

FROM node:20-slim AS build
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
LABEL org.opencontainers.image.title="hr-talent-evaluator-frontend" \
      org.opencontainers.image.description="Candidate App + Admin Dashboard"

RUN apk add --no-cache gettext

COPY --from=build /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/50-inject-env.sh
RUN chmod +x /docker-entrypoint.d/50-inject-env.sh

ENV API_BASE_URL=""

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1
