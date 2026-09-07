"""
TT Simulator API Server — FastAPI + WebSocket execution gateway.

POST /execute  — synchronous, returns full stdout/stderr as JSON
WS   /execute  — streams stdout/stderr chunks as JSON messages

Backends:
  ttlang-sim   — pure Python, runs via `ttlang-sim <file>`
  ttsim-wh     — Wormhole hardware-emulation binary (TT_METAL_SIMULATOR env)
  ttsim-bh     — Blackhole hardware-emulation binary

Auth: X-API-Key header checked against comma-separated API_KEYS env var.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEYS_RAW = os.environ.get("API_KEYS", "")
VALID_API_KEYS: set[str] = {k.strip() for k in API_KEYS_RAW.split(",") if k.strip()}

EXEC_TIMEOUT_SECS = int(os.environ.get("EXEC_TIMEOUT", "30"))
# SIM_HOME/<chip>/libttsim_<chip>.so + SIM_HOME/<chip>/soc_descriptor.yaml -- one
# subdirectory per chip, since ttsim resolves the descriptor as a sibling of the
# .so file, and wh/bh can't share a directory without clobbering each other's
# descriptor.
SIM_HOME = Path(os.environ.get("SIM_HOME", Path.home() / "sim"))
# Python interpreter with ttnn importable (the tt-metal python_env), NOT the
# interpreter running this API server -- the server itself only needs
# fastapi/uvicorn and never imports ttnn.
TT_METAL_PYTHON = os.environ.get("TT_METAL_PYTHON", sys.executable)
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "")
# Extra dir(s) for LD_LIBRARY_PATH (e.g. Tenstorrent's ULFM OpenMPI build),
# colon-separated.
TT_EXTRA_LD_LIBRARY_PATH = os.environ.get("TT_EXTRA_LD_LIBRARY_PATH", "")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("tt-sim-api")

app = FastAPI(title="TT Simulator API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Backend(str, Enum):
    ttlang_sim = "ttlang-sim"
    ttsim_wh = "ttsim-wh"
    ttsim_bh = "ttsim-bh"


class ExecuteRequest(BaseModel):
    code: str = Field(..., description="Python source code to execute")
    backend: Backend = Field(Backend.ttlang_sim, description="Execution backend")
    timeout: int = Field(EXEC_TIMEOUT_SECS, ge=1, le=300, description="Timeout in seconds")


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(x_api_key: str | None) -> None:
    """Raise 401 if API keys are configured and the header doesn't match."""
    if not VALID_API_KEYS:
        return  # auth disabled
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _build_cmd(backend: Backend, script_path: str) -> list[str]:
    """Return the command list for the given backend."""
    if backend == Backend.ttlang_sim:
        ttlang_sim = shutil.which("ttlang-sim")
        if not ttlang_sim:
            raise HTTPException(status_code=503, detail="ttlang-sim not found in PATH")
        return [ttlang_sim, script_path]

    if backend in (Backend.ttsim_wh, Backend.ttsim_bh):
        chip = "wh" if backend == Backend.ttsim_wh else "bh"
        so_path = SIM_HOME / chip / f"libttsim_{chip}.so"
        if not so_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"ttsim binary not found at {so_path}. Run the dev-container setup first.",
            )
        if not TT_METAL_HOME:
            raise HTTPException(status_code=503, detail="TT_METAL_HOME not configured")
        # TT_METAL_PYTHON may be an absolute path or a bare command resolved
        # via PATH (e.g. "python3") -- shutil.which() handles both; it
        # returns absolute paths unchanged if they're executable.
        resolved_python = shutil.which(TT_METAL_PYTHON)
        if not resolved_python:
            raise HTTPException(
                status_code=503, detail=f"TT_METAL_PYTHON not found: {TT_METAL_PYTHON}"
            )
        return [resolved_python, script_path]

    raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")


def _build_env(backend: Backend) -> dict[str, str]:
    """Return extra environment variables needed by the backend."""
    env = os.environ.copy()
    if backend in (Backend.ttsim_wh, Backend.ttsim_bh):
        chip = "wh" if backend == Backend.ttsim_wh else "bh"
        env["TT_METAL_HOME"] = TT_METAL_HOME
        env["TT_METAL_SIMULATOR"] = str(SIM_HOME / chip / f"libttsim_{chip}.so")
        env.setdefault("TT_METAL_ARCH_NAME", "wormhole_b0" if chip == "wh" else "blackhole")
        env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
        env["TT_METAL_DISABLE_SFPLOADMACRO"] = "1"
        pythonpath = [TT_METAL_HOME, str(Path(TT_METAL_HOME) / "ttnn")]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath)
        if TT_EXTRA_LD_LIBRARY_PATH:
            ld = [TT_EXTRA_LD_LIBRARY_PATH]
            if env.get("LD_LIBRARY_PATH"):
                ld.append(env["LD_LIBRARY_PATH"])
            env["LD_LIBRARY_PATH"] = ":".join(ld)
    return env


# ---------------------------------------------------------------------------
# Async subprocess streaming
# ---------------------------------------------------------------------------

async def _stream_output(
    backend: Backend,
    code: str,
    timeout: int,
) -> AsyncIterator[dict]:
    """
    Execute code in a temp file and yield JSON-serialisable dicts:
      {"type": "stdout", "data": "<chunk>"}
      {"type": "stderr", "data": "<chunk>"}
      {"type": "exit",   "code": <int>}
    """
    # tt-metal writes JIT kernel-build artifacts to `generated/` relative to
    # the process cwd, so each run gets its own writable working directory.
    workdir = tempfile.mkdtemp(prefix="ttsim_run_")
    script_path = str(Path(workdir) / "script.py")
    with open(script_path, "w") as f:
        f.write(code)

    try:
        cmd = _build_cmd(backend, script_path)
        env = _build_env(backend)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=workdir,
        )

        async def _read_stream(stream: asyncio.StreamReader, kind: str):
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                yield {"type": kind, "data": chunk.decode("utf-8", errors="replace")}

        async def _collect():
            async for msg in _read_stream(proc.stdout, "stdout"):
                yield msg
            async for msg in _read_stream(proc.stderr, "stderr"):
                yield msg

        try:
            async with asyncio.timeout(timeout):
                async for msg in _collect():
                    yield msg
                await proc.wait()
        except TimeoutError:
            proc.kill()
            await proc.wait()
            yield {"type": "stderr", "data": f"\n[TIMEOUT after {timeout}s]\n"}
            yield {"type": "exit", "code": -1}
            return

        yield {"type": "exit", "code": proc.returncode}

    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

@app.post("/execute", response_model=ExecuteResponse)
async def execute_sync(
    req: ExecuteRequest,
    x_api_key: str | None = Header(default=None),
) -> ExecuteResponse:
    """Run code synchronously and return full output."""
    _check_auth(x_api_key)
    log.info("POST /execute backend=%s len=%d", req.backend, len(req.code))

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    exit_code = 0

    async for msg in _stream_output(req.backend, req.code, req.timeout):
        if msg["type"] == "stdout":
            stdout_parts.append(msg["data"])
        elif msg["type"] == "stderr":
            stderr_parts.append(msg["data"])
        elif msg["type"] == "exit":
            exit_code = msg["code"]

    return ExecuteResponse(
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        exit_code=exit_code,
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/execute")
async def execute_ws(websocket: WebSocket) -> None:
    """Stream execution output over WebSocket.

    Client sends one JSON message:
      {"code": "...", "backend": "ttlang-sim", "timeout": 30, "api_key": "..."}

    Server sends multiple JSON messages (same shapes as _stream_output yields),
    terminated by an {"type": "exit", "code": <int>} message.
    """
    await websocket.accept()
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
    except asyncio.TimeoutError:
        await websocket.close(code=4008, reason="Initial message timeout")
        return

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_text(json.dumps({"type": "error", "data": "Invalid JSON"}))
        await websocket.close(code=4003, reason="Bad JSON")
        return

    # Auth
    api_key = payload.get("api_key")
    if VALID_API_KEYS and api_key not in VALID_API_KEYS:
        await websocket.send_text(json.dumps({"type": "error", "data": "Unauthorized"}))
        await websocket.close(code=4001, reason="Unauthorized")
        return

    code = payload.get("code", "")
    backend_raw = payload.get("backend", "ttlang-sim")
    timeout = min(int(payload.get("timeout", EXEC_TIMEOUT_SECS)), 300)

    try:
        backend = Backend(backend_raw)
    except ValueError:
        await websocket.send_text(
            json.dumps({"type": "error", "data": f"Unknown backend: {backend_raw}"})
        )
        await websocket.close(code=4003, reason="Bad backend")
        return

    log.info("WS /execute backend=%s len=%d", backend, len(code))

    try:
        async for msg in _stream_output(backend, code, timeout):
            await websocket.send_text(json.dumps(msg))
    except WebSocketDisconnect:
        log.info("WS client disconnected mid-stream")
    except HTTPException as exc:
        await websocket.send_text(
            json.dumps({"type": "error", "data": exc.detail})
        )
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def _ttsim_backend_ready(chip: str) -> bool:
    """True only if this backend could actually execute: the .so exists AND
    TT_METAL_HOME is configured AND TT_METAL_PYTHON resolves. The playground
    UI uses /health to decide what's runnable, so checking just the .so
    would report a backend as available when _build_cmd would 503 it."""
    so_exists = (SIM_HOME / chip / f"libttsim_{chip}.so").exists()
    return bool(so_exists and TT_METAL_HOME and shutil.which(TT_METAL_PYTHON))


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "backends": {
            "ttlang-sim": bool(shutil.which("ttlang-sim")),
            "ttsim-wh": _ttsim_backend_ready("wh"),
            "ttsim-bh": _ttsim_backend_ready("bh"),
        },
    }


# ---------------------------------------------------------------------------
# Entry point (local dev: `python api_server.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="info")
