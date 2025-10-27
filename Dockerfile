FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "${MPLCONFIGDIR}"

# Copy project metadata first (leverages Docker layer caching)
COPY pyproject.toml setup.py requirements.txt README.md ./

# Copy source and resources
COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY scripts ./scripts
COPY examples ./examples
COPY tests ./tests

ARG INSTALL_EXTRAS=false

RUN pip install --upgrade pip \
    && if [ "${INSTALL_EXTRAS}" = "true" ]; then \
           pip install ".[dev]"; \
       else \
           pip install .; \
       fi

CMD ["python", "scripts/run_full_pipeline.py", "--demo", "--kappa-method", "acoustic_exact", "--graybody", "acoustic_wkb"]

