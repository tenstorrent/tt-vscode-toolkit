FROM ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest

# Install Python API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    pydantic

# Copy server source
COPY api_server.py /opt/api/api_server.py
WORKDIR /opt/api

# SIM_HOME: directory containing libttsim_*.so files (mounted at runtime)
ENV SIM_HOME=/sim
ENV EXEC_TIMEOUT=30

EXPOSE 8080

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8080"]
