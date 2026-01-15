# Tenstorrent VSCode Toolkit - Docker Image
# Based on code-server for browser-based VSCode experience
# Includes Tenstorrent extension preinstalled

FROM codercom/code-server:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Switch to root for installation
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create coder user home directory structure
RUN mkdir -p /home/coder/.local/share/code-server/extensions \
    && mkdir -p /home/coder/tt-scratchpad \
    && mkdir -p /home/coder/models

# Copy the extension package into the container
# This expects the .vsix file to be built before docker build
COPY tt-vscode-toolkit-*.vsix /tmp/extension.vsix

# Copy entrypoint script
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Install the extension
RUN code-server --install-extension /tmp/extension.vsix \
    && rm /tmp/extension.vsix \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Set proper permissions
RUN chown -R coder:coder /home/coder

# Switch back to coder user
USER coder

# Set working directory
WORKDIR /home/coder

# Expose code-server port
EXPOSE 8080

# Set default password (override with env var PASSWORD)
ENV PASSWORD=tenstorrent

# Use custom entrypoint with helpful logging
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
