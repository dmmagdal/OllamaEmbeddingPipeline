FROM ghcr.io/ggml-org/llama.cpp:full

# Override llama.cpp entrypoint.
ENTRYPOINT []

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
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential

#######################################################################
# Conda/Mamba Setup.
#######################################################################

# Install Mambaforge (Mamba + Conda).
RUN wget https://github.com/conda-forge/miniforge/releases/download/4.12.0-0/Mambaforge-4.12.0-0-Linux-x86_64.sh -O /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm -f /tmp/mambaforge.sh

# Set environment variables for conda.
ENV PATH=/opt/conda/bin:$PATH

# Copy env.yml file (no cuda) into the container.
COPY env.yml /app/env.yml

# Copy config.json file.
COPY config.json /app/config.json

# Create the conda environment from the env.yml.
RUN mamba env create -f /app/env.yml

#######################################################################
# Llama.cpp Setup.
#######################################################################

# Clone the llama.cpp repo from GitHub.
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /app/llama.cpp

# Build the llama.cpp tools.
RUN cd /app/llama.cpp && \
    cmake -B build && \ 
    cmake --build build --config Release && \ 
    cd ..

# Add llama.cpp binaries to PATH.
ENV PATH=/opt/llama.cpp/build/bin:$PATH

#######################################################################
# Ollama Setup.
#######################################################################

# Install Ollama CLI.
RUN curl -sSL https://ollama.com/install.sh | bash

# Ensure it's in PATH.
ENV PATH=/root/.ollama/bin:$PATH

# Start Ollama server.
# RUN ollama serve

#######################################################################
# Runtime.
#######################################################################

# Set the working directory to be /app.
WORKDIR /app

# Activate the environment and install dependencies (if any).
RUN echo "conda activate ollama-pipeline" >> ~/.bashrc

# Install any additional dependencies (if needed)
RUN /opt/conda/bin/mamba install -y -n ollama-pipeline pip

# Run pipeline.
# CMD ["python3", "/app/pipeline.py"]                                                        # Can't find command python3
# CMD ["python", "/app/pipeline.py"]                                                         # Can't find command python
# CMD ["/opt/conda/bin/mamba", "run", "-n", "ollama-pipeline", "python", "/app/pipeline.py"] # Can't find command mamba
# CMD ["/opt/conda/bin/conda", "run", "-n", "ollama-pipeline", "python", "/app/pipeline.py"] # Error/return 0

# CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate ollama-pipeline &&", "python", "/app/pipeline.py"]
# CMD ["/bin/sh", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate ollama-pipeline &&", "python", "/app/pipeline.py"]

# CMD ["python", "/app/pipeline.py"]
# CMD ["conda", "run", "-n", "ollama-pipeline", "python", "/app/pipeline.py"] # Runs but requires ollama server to be running
# CMD ["ollama serve & sleep 5 &&" "conda", "run", "-n", "ollama-pipeline", "python", "/app/pipeline.py"]
CMD bash -c "ollama serve & sleep 5 && conda run -n ollama-pipeline python /app/pipeline.py"