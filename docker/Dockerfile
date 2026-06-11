FROM docker.io/pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    neovim \
    wget \
    curl \
    git \
    zsh \
    libxcb-xinerama0 \
    lsb-release \
    bat \
    ripgrep \
    ca-certificates \
    gnupg \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install GitHub cli
RUN mkdir -p -m 755 /etc/apt/keyrings \
  && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
     | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
  && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
     | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
  && apt-get update \
  && apt-get install -y --no-install-recommends gh \
  && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" slurm

# allow pip install of "sklearn", which should be replaced by "scikit-learn"
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

SHELL ["/bin/bash", "--login", "-c"]

RUN conda install -y jupyter \
  && conda clean -a -y

# Set up environment. Keep conda's bin first so python/pip and the baked-in
# humite-train / humite-generate console scripts resolve without activation.
ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin


RUN ["/bin/bash", "-c", "source /opt/conda/bin/activate"]

# --------------------------------------------------------------------------- #
# Bake the humite source into the image for self-contained, reproducible runs.
# Declared late so the build-args (which change every build) don't bust the
# heavy apt/pip dependency layers above.
ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="humite" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/uhh-pd-ml/SPADE"

# Dependencies are already pinned in requirements.txt, so install the package
# without re-resolving them (--no-deps). This installs the humite-train /
# humite-generate console scripts and ships the Hydra YAML configs. A mounted
# working copy on PYTHONPATH still overrides this baked-in install for dev.
COPY . /opt/humite
RUN pip install --no-cache-dir --no-deps /opt/humite
