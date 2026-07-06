FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    OPENDDE_ROOT_DIR=/opendde_data

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        hmmer \
        kalign \
        libglib2.0-0 \
        libgl1 \
        libxrender1 \
        pkg-config \
        python-is-python3 \
        python3 \
        python3-dev \
        tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . .
RUN uv pip install --system --break-system-packages --torch-backend cu126 -e '.[gpu]' && \
    opendde --help >/dev/null

VOLUME ["/opendde_data"]
CMD ["opendde", "--help"]
