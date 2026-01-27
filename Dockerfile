# Tenstorrent VSCode Toolkit - Docker Image
# Based on code-server for browser-based VSCode experience
# Includes Tenstorrent extension preinstalled

FROM codercom/code-server:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Switch to root for installation
USER root

# Install system dependencies including Node.js for Claude CLI
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install HuggingFace CLI and matplotlib for cookbook examples
# Note: --break-system-packages is safe in containers (PEP 668)
RUN pip3 install --no-cache-dir --break-system-packages huggingface-hub[cli] matplotlib

# Install Claude Code CLI for AI-assisted development
# Authentication via ANTHROPIC_API_KEY environment variable
RUN npm install -g @anthropic-ai/claude-code

# Create coder user home directory structure
RUN mkdir -p /home/coder/.local/share/code-server/extensions \
    && mkdir -p /home/coder/tt-scratchpad \
    && mkdir -p /home/coder/models

# Copy the extension package into the container
# This expects the .vsix file to be built before docker build
COPY tt-vscode-toolkit-*.vsix /tmp/extension.vsix

# Copy entrypoint script
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Install the extension and configure VSCode settings
RUN code-server --install-extension /tmp/extension.vsix \
    && rm /tmp/extension.vsix \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Configure code-server with Tenstorrent theme and terminal login shell
# Terminal login shell ensures .bashrc is sourced (MOTD displays correctly)
RUN mkdir -p /home/coder/.local/share/code-server/User && \
    echo '{\n\
  "workbench.colorTheme": "Tenstorrent Dark",\n\
  "workbench.tips.enabled": false,\n\
  "telemetry.telemetryLevel": "off",\n\
  "update.mode": "none",\n\
  "terminal.integrated.defaultProfile.linux": "bash",\n\
  "terminal.integrated.profiles.linux": {\n\
    "bash": {\n\
      "path": "bash",\n\
      "args": ["-l"],\n\
      "icon": "terminal-bash"\n\
    }\n\
  }\n\
}' > /home/coder/.local/share/code-server/User/settings.json

# Set proper permissions
RUN chown -R coder:coder /home/coder

# Switch back to coder user
USER coder

# Set working directory
WORKDIR /home/coder

# Expose code-server port
EXPOSE 8080

# For security, no default password is set in the image.
# You MUST provide a strong PASSWORD environment variable at runtime, for example:
#   docker run -e PASSWORD="$(openssl rand -base64 32)" ...
# If PASSWORD is not set, the entrypoint or application should refuse to start.

# Use custom entrypoint with helpful logging
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
