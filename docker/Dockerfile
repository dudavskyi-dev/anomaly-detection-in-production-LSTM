# Multi-stage build (P08 deliverable #6): the builder stage has pip/setuptools/wheel and
# whatever it takes to resolve dependencies; none of that reaches the final image, which only
# ever gets the already-built site-packages tree copied in. See docs/decisions/P08-serving.md
# for the measured image-size difference this makes.
#
# Ships PyTorch bundles only by default (the `.[torch,serving]` extras) — TensorFlow is
# deliberately not installed here to keep the image lean; serving a TensorFlow bundle would need
# rebuilding this image with `.[tf,serving]` instead (or both), a real, documented scope choice,
# not an oversight. `EXTRAS` is a build arg (not hardcoded) specifically so P09's drift-monitor
# service (docker/docker-compose.yml) can reuse this same Dockerfile with
# `.[torch,serving,tracking,monitoring]` instead of maintaining a near-duplicate second
# Dockerfile just to add MLflow + Prometheus-client — the default below is unchanged, so
# `docker build` with no `--build-arg` still produces exactly the image
# docs/decisions/P08-serving.md's measurements are about.

FROM python:3.11-slim AS builder

ARG EXTRAS=torch,serving

WORKDIR /build

COPY pyproject.toml README.md ./

# Install the CPU-only torch wheel first: the default PyPI `torch` wheel pulls in ~2GB of
# NVIDIA CUDA runtime libraries (nvidia_cusolver, nvidia_cusparse, ...) that a serving container
# with no GPU never uses — real, measured build failures here included pip timing out / DNS
# errors part-way through those huge downloads. Once a `torch~=2.14.0` satisfying this index is
# installed, the follow-up `pip install ".[$EXTRAS]"` sees the constraint already met and does
# not reinstall it with the CUDA-bundled build.
#
# `COPY pdm ./pdm` is deliberately *after* this, not before: this step depends only on
# pyproject.toml (which names the `torch` extra's version constraint), never on the `pdm`
# package's own source — putting it first means an ordinary source-code change (P10's own
# instrumentation work, say) no longer invalidates this ~200MB download's build cache, which a
# `COPY pdm ./pdm` placed before it would do on every single edit.
RUN pip install --no-cache-dir --prefix=/install torch~=2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY pdm ./pdm
RUN PYTHONPATH=/install/lib/python3.11/site-packages pip install --no-cache-dir --prefix=/install ".[$EXTRAS]"


FROM python:3.11-slim

# Non-root: a compromised inference process should not run as root.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY pdm ./pdm
COPY pyproject.toml README.md ./

RUN chown -R appuser:appuser /app
USER appuser

ENV PDM__SERVING__HOST=0.0.0.0 \
    PDM__SERVING__PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "pdm.cli", "serve"]
