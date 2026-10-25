# The service image: publisher, processor and map server share it
# (compose.yaml picks the command). Only the pipeline dependency group is
# installed, from uv.lock exactly, and only the committed files the
# service reads are copied in; no account or network is needed at run time.

FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM python:3.14.8-slim-trixie

COPY --from=uv /uv /bin/uv

# The system interpreter matches requires-python; never download another.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --group pipeline

COPY src src
COPY scripts scripts
COPY data/extract data/extract
COPY data/reference data/reference
COPY results/coverage/tuning_coverage.parquet results/coverage/

# The map page and the closures it draws, served beside the run's volume.
COPY src/cloud/map/index.html /srv/
COPY data/reference/us/closures.geojson /srv/

# A named volume mounted at /demo takes this directory's owner on first
# use, so the service can run unprivileged.
RUN useradd --uid 1000 --create-home app && mkdir /demo && chown app /demo
USER app
