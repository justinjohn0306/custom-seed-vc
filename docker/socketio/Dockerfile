FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_CACHE_DIR="/opt/poetry/cache"
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy poetry configuration files
COPY pyproject.toml poetry.lock poetry.toml ./
# Copy README.md (required by poetry for package installation)
COPY README.md ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main,client --without dev --no-root

# Copy source code
COPY seed_vc/ ./seed_vc/
COPY configs/ ./configs/
COPY assets/ ./assets/

# Create cache directory for HuggingFace models
RUN mkdir -p /root/.cache/huggingface

# Expose port
EXPOSE 5000

# Set environment variables with defaults
ENV LOG_LEVEL=INFO

# Command to run the server
CMD poetry run python -m seed_vc.socketio.server --host 0.0.0.0 --port 5000 --log-level $LOG_LEVEL
