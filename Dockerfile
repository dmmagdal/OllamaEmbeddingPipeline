FROM ghcr.io/ggml-org/llama.cpp:full

# Copy over pipeline script.
COPY pipeline.py /app/pipeline.py

# Install python3 and pip.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# Install Mambaforge (Mamba + Conda).
RUN wget https://github.com/conda-forge/miniforge/releases/download/4.12.0-0/Mambaforge-4.12.0-0-Linux-x86_64.sh -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm -f /tmp/mambaforge.sh

# Set environment variables for conda.
ENV PATH=/opt/conda/bin:$PATH

# Copy env.yml file (no cuda) into the container.
COPY env.yml /app/env.yml

# Create the conda environment from the env.yml.
RUN mamba env create -f /app/env.yml

# Activate the environment and install dependencies (if any).
RUN echo "conda activate ollama-pipeline" >> ~/.bashrc

# Run pipeline.
CMD ["python3", "/app/pipeline.py"]