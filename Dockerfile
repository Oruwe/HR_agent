# =============================================================================
# hr-talent-evaluator -- production image
#
# Two design points worth noting:
#
# 1. The Moss retrieval runtime ships *inside* this image. There is no retrieval
#    tier to deploy alongside it and no vector database to operate, which is
#    what makes the 10ms retrieval budget a property of the process rather than
#    of somebody's network.
# 2. The default command is the offline demo. An image that does something
#    useful with no credentials is an image people actually run.
# =============================================================================

# ---- build stage ------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Core dependencies only. The voice/cognition/telemetry extras are large and
# every one of them is optional at runtime -- the agent degrades rather than
# failing to boot -- so they are installed by deployment, not baked in here.
COPY pyproject.toml README.md ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install "pydantic>=2.6" "PyYAML>=6.0" "numpy>=1.26"

# ---- runtime stage ----------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="hr-talent-evaluator" \
      org.opencontainers.image.description="Sub-160ms voice HR screening agent with Moss retrieval" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HRTE_ENV=production \
    HRTE_PII_MODE=strict

# Unprivileged user. A screening worker handles candidate audio; root is not a
# risk worth taking for the convenience.
RUN useradd --create-home --uid 10001 screener

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=screener:screener agent.yaml SOUL.md EXPLAINABILITY.md README.md ./
COPY --chown=screener:screener app ./app
COPY --chown=screener:screener scripts ./scripts
COPY --chown=screener:screener docs ./docs

USER screener

# Compliance self-check: fails the container if the OpenGAP manifest, the
# latency budget, or the PII invariant is broken.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-m", "app.main", "verify"]

ENTRYPOINT ["python", "-m", "app.main"]
CMD ["demo"]
