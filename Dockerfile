# Use Python 3.11 slim image
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen ensures we use the exact versions from uv.lock
# --no-install-project skips installing the project itself (since it's just scripts)
# --no-dev skips development dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Create symlinks to /workspace for persistent storage
# We remove existing directories (if copied) and link to /workspace
RUN rm -rf checkpoints samples wandb && \
    mkdir -p /workspace/checkpoints /workspace/samples /workspace/wandb && \
    ln -s /workspace/checkpoints /app/checkpoints && \
    ln -s /workspace/samples /app/samples && \
    ln -s /workspace/wandb /app/wandb

# Define volumes for persistent data
VOLUME ["/workspace/checkpoints", "/workspace/samples", "/workspace/wandb"]

# Run the training script
# We use `accelerate launch` to handle distributed training
CMD ["uv", "run", "accelerate", "launch", "--config_file", "accelerate_config.yaml", "train.py", "--log"]