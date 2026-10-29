/**
 * Terminal Commands Configuration
 *
 * This file is the single source of truth for all terminal commands used in the walkthrough.
 * Commands defined here are:
 * 1. Shown to users in markdown lessons (so they know what will execute)
 * 2. Executed by the extension (ensuring what's shown matches what runs)
 *
 * When updating commands, update them here and they'll automatically sync everywhere.
 */

/**
 * Command template that can include variables to be replaced at runtime
 */
export interface CommandTemplate {
  /** Unique identifier for this command */
  id: string;

  /** Display name for the command */
  name: string;

  /** The actual command template (may include {{variables}}) */
  template: string;

  /** Description of what this command does */
  description: string;

  /** Variables that will be replaced in the template */
  variables?: string[];
}

// ── Forge / XLA shared Python snippets ──────────────────────────────────────
// Sent as `python3 -c "..."` — no heredoc, so they work in bash, zsh, and fish.
// tt_torch must be imported before jax so the TT PJRT plugin registers TT
// hardware with JAX; without it jax.devices() falls back silently to CPU.
// Templates that use these shared snippet constants should follow:
// `source ~/tt-forge-venv/bin/activate && python3 -c "\n${SNIPPET}\n"`

/** forge activation probe — used by BUILD_FORGE_FROM_SOURCE, INSTALL_FORGE */
const FORGE_ACTIVATION_PY =
  "try:\n    import tt_torch  # pre-loads TT PJRT plugin; jax.devices() sees TT hardware\n" +
  "except ImportError:\n    pass\n" +
  "try:\n    import forge; forge_info = 'forge ' + forge.__version__\n" +
  "except ImportError:\n    import tt_torch; forge_info = 'tt-forge (tt_torch API)'\n" +
  "import jax\n" +
  "print('forge/tt-forge:', forge_info)\n" +
  "print('jax           :', jax.__version__)\n" +
  "print('devices       :', jax.devices())";

/** forge + torch-xla full version probe — TEST_FORGE_INSTALL, TEST_FORGE_INSTALL_WHEEL, VERIFY_FORGE_STACK */
const FORGE_FULL_VERIFY_PY =
  "try:\n    import tt_torch  # pre-loads TT PJRT plugin; jax.devices() sees TT hardware\n" +
  "except ImportError:\n    pass\n" +
  "try:\n    import forge; forge_info = 'forge     : ' + forge.__version__\n" +
  "except ImportError:\n    import tt_torch; forge_info = 'tt-forge  : tt_torch API'\n" +
  "import jax, torch_xla\n" +
  "print(forge_info)\n" +
  "print('jax       :', jax.__version__)\n" +
  "print('torch_xla :', torch_xla.__version__)\n" +
  "print('tt devices:', jax.devices())";

/** minimal JAX device listing — ACTIVATE_FORGE_ENV, INSTALL_TT_XLA */
const JAX_DEVICE_CHECK_PY =
  "try:\n    import tt_torch  # pre-loads TT PJRT plugin; jax.devices() sees TT hardware\n" +
  "except ImportError:\n    pass\n" +
  "import jax\n" +
  "print('TT devices:', jax.devices())";

/**
 * ttsim QEMU Bridge architecture:
 * ttsim-qemu is a QEMU fork (github.com/tenstorrent/ttsim-qemu) that adds a
 * `ttsim` PCI device (vendor 0x1e52) to any Linux VM. The user brings their own
 * base image (Ubuntu 24.04 cloud image); libttsim_wh.so runs on the host and is
 * bridged to the guest via the PCI device. Inside the VM, TT-Metal sees real
 * Wormhole hardware — no TT_METAL_SIMULATOR env var needed.
 * Constraint: slow dispatch recommended (TT_METAL_SLOW_DISPATCH_MODE=1) — fast
 * dispatch works but its determinism under simulation is uncharacterized.
 */

/**
 * All terminal commands used in the walkthrough
 */
export const TERMINAL_COMMANDS: Record<string, CommandTemplate> = {
  // ========================================
  // Set Up: Where Am I? / Install and Upgrade with tt
  // ========================================
  // Templates are the bare command the reader types. Environment guards live in the
  // handler (src/utils/environment.ts), never here, so the docs site prints the
  // command as written.

  SETUP_WHERE_AM_I: {
    id: 'setup-where-am-i',
    name: 'Where am I?',
    template: 'echo "uv:       $(uv --version 2>/dev/null || echo none)"; echo "tt:       $(which -a tt 2>/dev/null | tr \'\\n\' \' \')"; echo "chips:    $(tt-smi -s 2>/dev/null | python3 -c \'import sys,json; print(len(json.load(sys.stdin)["device_info"]))\' 2>/dev/null || echo 0)"; echo "tt-metal: $([ -d ~/tt-metal ] && echo ~/tt-metal || echo none)"; echo "docker:   $(docker --version 2>/dev/null || echo none)"',
    description: 'Reports uv, tt, chip count, tt-metal source and Docker so the reader knows which lessons apply',
  },

  SETUP_INSTALL_TT: {
    id: 'setup-install-tt',
    name: 'Install tt',
    template: 'uv tool install --python 3.12 tenstorrent',
    description: 'Installs the tt CLI (PyPI package "tenstorrent") as a uv tool on Python 3.12 (3.10 resolves an old release)',
  },

  SETUP_USE_INSTALLED_TOOLS: {
    id: 'setup-use-installed-tools',
    name: "Use the pod's tt-smi and tt-flash",
    template: 'tt config set tools.override.tt-smi ~/.tt-installer-venv/bin/tt-smi && tt config set tools.override.tt-flash ~/.tt-installer-venv/bin/tt-flash',
    description: 'Cloud pod: points tt at the tt-smi/tt-flash tt-installer already put in ~/.tt-installer-venv, since tt update is not for the pod',
  },

  SETUP_DEVICE_STATUS: {
    id: 'setup-device-status',
    name: 'Device status',
    template: 'tt --version && tt device status',
    description: 'One row per chip: board, bus id, temperature, power, clock, DRAM state',
  },

  SETUP_DEVICE_INFO: {
    id: 'setup-device-info',
    name: 'Device info',
    template: 'tt device info',
    description: 'Board ids, PCIe speed and width, firmware versions, one column per chip',
  },

  SETUP_UPDATE_DRY_RUN: {
    id: 'setup-update-dry-run',
    name: 'Preview upgrade',
    template: 'tt update --dry-run',
    description: 'Shows the golden-set update plan without changing anything (safe on the cloud pod)',
  },

  SETUP_UPDATE: {
    id: 'setup-update',
    name: 'Upgrade to the golden set',
    template: 'tt update',
    description: 'Installs or upgrades tools, driver, firmware and HugePages to the tested golden set (guarded on the cloud pod)',
  },

  SETUP_SELF_UPDATE: {
    id: 'setup-self-update',
    name: 'Upgrade tt',
    template: 'tt self update --check && tt self update',
    description: 'Upgrades the tt command itself to the newest release',
  },

  SETUP_VERIFY_TTNN_CLOUD: {
    id: 'setup-verify-ttnn-cloud',
    name: 'Check TT-NN (Python env)',
    template: 'python3 -c "import ttnn; print(\'✓ TTNN\', getattr(ttnn, \'__version__\', \'(source build)\'), ttnn.__file__)"',
    description: 'Imports ttnn in the active Python environment (cloud pod: ~/tt-metal/python_env)',
  },

  SETUP_VERIFY_TTNN_QB2: {
    id: 'setup-verify-ttnn-qb2',
    name: 'Check TT-NN (container)',
    template: 'tt-metalium -c "python3 -c \'import ttnn; print(getattr(ttnn, \\"__version__\\", \\"import OK\\"))\'"',
    description: 'Imports ttnn inside the tt-metalium container (TT-QuietBox 2)',
  },

  // ========================================
  // Lesson 1: Hardware Detection
  // ========================================

  TT_SMI: {
    id: 'tt-smi',
    name: 'Hardware Detection',
    template: 'tt-smi',
    description: 'Scans for connected Tenstorrent devices and displays their status',
  },

  // ========================================
  // Lesson 2: Verify Installation
  // ========================================

  VERIFY_INSTALLATION: {
    id: 'verify-installation',
    name: 'Verify TT-Metal Installation',
    template: 'python3 -m ttnn.examples.usage.run_op_on_device',
    description: 'Runs a test operation to verify tt-metal is working correctly',
  },




  // Hugging Face Authentication
  SET_HF_TOKEN: {
    id: 'set-hf-token',
    name: 'Set Hugging Face Token',
    template: 'export HF_TOKEN="{{token}}"',
    description: 'Sets your Hugging Face access token as an environment variable',
    variables: ['token'],
  },

  LOGIN_HF: {
    id: 'login-hf',
    name: 'Login to Hugging Face',
    template: 'tt model login',
    description: 'Authenticates with Hugging Face using your token',
  },

  // Model Download

  // Clone TT-Metal
  CLONE_TT_METAL: {
    id: 'clone-tt-metal',
    name: 'Clone TT-Metal Repository',
    template: 'git clone https://github.com/tenstorrent/tt-metal.git "{{path}}" --recurse-submodules',
    description: 'Clones the tt-metal repository with all submodules',
    variables: ['path'],
  },

  // Setup Environment
  SETUP_ENVIRONMENT: {
    id: 'setup-environment',
    name: 'Setup Python Environment',
    template:
      'cd "{{ttMetalPath}}" && export PYTHONPATH=$(pwd) && pip install --upgrade pip setuptools wheel && pip install -r tt_metal/python_env/requirements-dev.txt',
    description: 'Sets PYTHONPATH, upgrades pip/setuptools/wheel, and installs Python dependencies for tt-metal',
    variables: ['ttMetalPath'],
  },

  // Run Inference

  // Interactive Chat (Lesson 4)



  // HTTP API Server (Lesson 5)





  // ========================================
  // Serve Models: tt model / tt serve
  // ========================================

  MODEL_LIST: {
    id: 'model-list',
    name: 'Models for this hardware',
    template: 'tt model list',
    description: 'Models with a serving profile for the detected board',
  },

  MODEL_LIST_CATALOG: {
    id: 'model-list-catalog',
    name: 'Released catalog',
    template: 'tt model list --catalog',
    description: 'Only the released catalog Tenstorrent ships and tests through tt-inference-server',
  },

  MODEL_INFO: {
    id: 'model-info',
    name: 'Model info',
    template: 'tt model info Llama-3.1-8B-Instruct',
    description: 'What a model needs (disk, RAM) and its validation status per board',
  },

  MODEL_SEARCH: {
    id: 'model-search',
    name: 'Search the Hub',
    template: 'tt model search qwen3 --limit 5',
    description: 'Community tt-model bundles on the Hugging Face Hub matching a name (untested by Tenstorrent)',
  },

  MODEL_PULL: {
    id: 'model-pull',
    name: 'Pull weights',
    template: 'tt model pull Llama-3.1-8B-Instruct',
    description: 'Downloads the weights into the Hugging Face cache ahead of serving',
  },

  MODEL_SERVE_DRY_RUN_N150: {
    id: 'model-serve-dry-run-n150',
    name: 'Dry run: Llama-3.1-8B-Instruct',
    template: 'tt serve Llama-3.1-8B-Instruct --dry-run',
    description: 'Shows what tt serve would run for the one-chip model; starts nothing',
  },

  MODEL_SERVE_N150: {
    id: 'model-serve-n150',
    name: 'Serve Llama-3.1-8B-Instruct',
    template: 'tt serve Llama-3.1-8B-Instruct',
    description: 'Serves the one-chip model in a container (needs Docker; guarded on the cloud pod)',
  },

  MODEL_SERVE_DRY_RUN_QB2: {
    id: 'model-serve-dry-run-qb2',
    name: 'Dry run: Qwen3-32B',
    template: 'tt serve Qwen3-32B --dry-run',
    description: 'Shows what tt serve would run for the TT-QuietBox 2 model; starts nothing',
  },

  MODEL_SERVE_QB2: {
    id: 'model-serve-qb2',
    name: 'Serve Qwen3-32B',
    template: 'tt serve Qwen3-32B',
    description: 'Serves the model a TT-QuietBox 2 ships with (needs Docker; guarded on the cloud pod)',
  },

  MODEL_PS: {
    id: 'model-ps',
    name: 'What is running',
    template: 'tt model ps',
    description: 'Every server tt started: backend, container, port, health, uptime',
  },

  MODEL_LOGS: {
    id: 'model-logs',
    name: 'Server log',
    template: 'tt model logs',
    description: 'The newest server log (add --follow to tail it)',
  },

  MODEL_CURL: {
    id: 'model-curl',
    name: 'Ask the model',
    template: 'tt model curl "Explain what an API is in one sentence." --max-tokens 256',
    description: 'Sends one chat completion to the served model and prints the JSON reply',
  },

  MODEL_STOP: {
    id: 'model-stop',
    name: 'Stop the model',
    template: 'tt model stop',
    description: 'Stops the served model (name it when more than one is running)',
  },

  VLLM_SERVE_CLOUD_N150: {
    id: 'vllm-serve-cloud-n150',
    name: 'Serve Llama-3.1-8B-Instruct (vLLM, no container)',
    template: 'source ~/tt-metal/build/python_env_vllm/bin/activate && export PYTHONPATH=$HOME/tt-metal/tools:$HOME/tt-metal/ttnn:$PYTHONPATH && MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 --max-num-seqs 32 --block-size 64 --port 20000',
    description: "Cloud pod: serves from the pod's vLLM environment with the Tenstorrent plugin, same /v1 endpoint as tt serve",
  },

  VLLM_CURL_CLOUD: {
    id: 'vllm-curl-cloud',
    name: 'Ask the model (curl)',
    template: 'curl -s http://localhost:20000/v1/chat/completions -H "Content-Type: application/json" -d \'{"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "Explain what an API is in one sentence."}], "max_tokens": 256}\' | python3 -m json.tool',
    description: 'One chat completion against the no-container server',
  },

  // ========================================
  // Connect Software: tt launch (launch-software lesson)
  // ========================================
  // Templates stay bare tt commands; the cloud guard lives in the handlers.

  LAUNCH_LIST: {
    id: 'launch-list',
    name: 'What can connect',
    template: 'tt launch list',
    description: 'Every client tt can connect to the served model, and whether each is usable now',
  },

  LAUNCH_OPENWEBUI_DRY_RUN: {
    id: 'launch-openwebui-dry-run',
    name: 'Dry run: Open WebUI',
    template: 'tt launch openwebui --dry-run',
    description: 'Shows the model, endpoint and container tt launch would use for Open WebUI; changes nothing',
  },

  LAUNCH_OPENWEBUI: {
    id: 'launch-openwebui',
    name: 'Launch Open WebUI',
    template: 'tt launch openwebui',
    description: 'Runs Open WebUI in a container pointed at the served model (needs Docker; guarded on the cloud pod)',
  },

  LAUNCH_STOP_OPENWEBUI: {
    id: 'launch-stop-openwebui',
    name: 'Stop Open WebUI',
    template: 'tt launch stop openwebui',
    description: 'Stops the Open WebUI container and keeps its chats and settings (needs Docker; guarded on the cloud pod)',
  },

  LAUNCH_INSTALL_AIDER: {
    id: 'launch-install-aider',
    name: 'Install Aider',
    template: 'uv tool install aider-chat',
    description: 'Installs Aider as a uv tool; works on the cloud pod, which has uv and no Node.js',
  },

  LAUNCH_AIDER_DRY_RUN: {
    id: 'launch-aider-dry-run',
    name: 'Dry run: Aider',
    template: 'tt launch aider --dry-run',
    description: 'Shows the exact aider command tt launch would run against the served model',
  },

  LAUNCH_AIDER: {
    id: 'launch-aider',
    name: 'Launch Aider',
    template: 'tt launch aider',
    description: 'Starts Aider in this terminal, pointed at the served model',
  },

  LAUNCH_QWENCODE_DRY_RUN: {
    id: 'launch-qwencode-dry-run',
    name: 'Dry run: Qwen Code',
    template: 'tt launch qwencode --dry-run',
    description: 'Shows the exact qwen command tt launch would run against the served model',
  },

  LAUNCH_QWENCODE: {
    id: 'launch-qwencode',
    name: 'Launch Qwen Code',
    template: 'tt launch qwencode',
    description: 'Starts Qwen Code in this terminal, pointed at the served model (needs Node.js and @qwen-code/qwen-code)',
  },

  LAUNCH_STUDIO_N150: {
    id: 'launch-studio-n150',
    name: 'TT-Studio: Llama-3.1-8B-Instruct',
    template: 'tt serve Llama-3.1-8B-Instruct --studio',
    description: 'Serves the one-chip model through TT-Studio, which brings its own web UI on port 3000 (needs Docker; guarded on the cloud pod)',
  },

  LAUNCH_STUDIO_QB2: {
    id: 'launch-studio-qb2',
    name: 'TT-Studio: Qwen3-32B',
    template: 'tt serve Qwen3-32B --studio',
    description: 'Serves the TT-QuietBox 2 model through TT-Studio, which brings its own web UI on port 3000 (needs Docker; guarded on the cloud pod)',
  },

  // Agent lessons (qb2-local-agents, qb2-openclaw-assistant) keep their scripts on
  // port 8000, so these serve there instead of tt's default 20000. Tool calling and the
  // parser come from the model catalog; nothing to pass.
  MODEL_SERVE_QB2_AGENTS_QWEN: {
    id: 'model-serve-qb2-agents-qwen',
    name: 'Serve Qwen3-32B on port 8000',
    template: 'tt serve Qwen3-32B --port 8000',
    description: 'Serves Qwen3-32B on port 8000, where the agent lessons expect it (needs Docker; guarded on the cloud pod)',
  },

  MODEL_SERVE_QB2_AGENTS_LLAMA: {
    id: 'model-serve-qb2-agents-llama',
    name: 'Serve Llama-3.3-70B-Instruct on port 8000',
    template: 'tt serve Llama-3.3-70B-Instruct --port 8000',
    description: 'Serves Llama-3.3-70B-Instruct on port 8000, where the agent lessons expect it (needs Docker; guarded on the cloud pod)',
  },

  // ========================================
  // Build on the Stack: environment-aware first steps
  // ========================================

  METAL_RUN_ADD_TENSORS: {
    id: 'metal-run-add-tensors',
    name: 'Run the first TT-NN tutorial',
    template: 'cd ~/tt-metal && python3 ttnn/tutorials/basic_python/ttnn_add_tensors.py',
    description: 'Adds two tensors on the device with the tutorial script in ~/tt-metal (cloud pod: python_env is already active)',
  },

  TTLANG_SETUP_CLOUD: {
    id: 'ttlang-setup-cloud',
    name: 'Activate TT-Lang and copy the tutorials',
    template: 'source ~/tt-lang-venv/bin/activate && tt-lang-setup-tutorials -t ~/ttlang-tutorials',
    description: "Activates the pod's TT-Lang venv and copies its bundled tutorials to ~/ttlang-tutorials",
  },

  TTLANG_RUN_TUTORIAL_SIM: {
    id: 'ttlang-run-tutorial-sim',
    name: 'Run a TT-Lang tutorial on the simulator',
    template: 'source ~/tt-lang-venv/bin/activate && ttlang-sim ~/ttlang-tutorials/elementwise/step_4_multinode_grid_auto.py',
    description: 'Runs the elementwise tutorial on the functional simulator; no hardware is touched',
  },

  // ========================================
  // Lesson 6: Production Inference with tt-inference-server
  // ========================================

  VERIFY_INFERENCE_SERVER_PREREQS: {
    id: 'verify-inference-server-prereqs',
    name: 'Verify tt-inference-server Prerequisites',
    template: 'echo "=== Checking Prerequisites ===" && which docker && docker --version && ls ~/.local/lib/tt-inference-server/run.py && tt-smi -s | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f\'  board: {c[\"board_type\"]}\') for c in d[\"device_info\"]]" 2>/dev/null || tt-smi && echo "=== ✓ Prerequisites OK ==="',
    description: 'Verifies Docker is installed, tt-inference-server run.py exists, and hardware is detected',
  },

  START_TT_INFERENCE_SERVER: {
    id: 'start-tt-inference-server',
    name: 'Start tt-inference-server (auto-detect hardware)',
    // --tt-device omitted: run.py auto-detects via tt-smi. --no-auth skips JWT_SECRET requirement.
    template: 'cd ~/.local/lib/tt-inference-server && python3 run.py --model Llama-3.1-8B-Instruct --workflow server --docker-server --no-auth',
    description: 'Starts vLLM server via tt-inference-server; hardware is auto-detected by run.py via tt-smi (no --tt-device flag needed)',
  },

  START_TT_INFERENCE_SERVER_N150: {
    id: 'start-tt-inference-server-n150',
    name: 'Start tt-inference-server (N150)',
    template: 'cd ~/.local/lib/tt-inference-server && python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 --workflow server --docker-server --no-auth',
    description: 'Starts vLLM server for Llama-3.1-8B-Instruct on N150 (Wormhole, 64K context)',
  },

  START_TT_INFERENCE_SERVER_N300: {
    id: 'start-tt-inference-server-n300',
    name: 'Start tt-inference-server (N300)',
    template: 'cd ~/.local/lib/tt-inference-server && python3 run.py --model Llama-3.1-8B-Instruct --tt-device n300 --workflow server --docker-server --no-auth',
    description: 'Starts vLLM server for Llama-3.1-8B-Instruct on N300 (Wormhole dual-chip, 128K context)',
  },

  TEST_TT_INFERENCE_SERVER_SIMPLE: {
    id: 'test-tt-inference-server-simple',
    name: 'Test tt-inference-server (Simple)',
    template: 'curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \'{"model": "Llama-3.1-8B-Instruct", "prompt": "Explain what a Tenstorrent AI accelerator is in one sentence.", "max_tokens": 50, "temperature": 0.7}\'',
    description: 'Tests the vLLM server started by tt-inference-server with OpenAI-compatible API',
  },

  TEST_TT_INFERENCE_SERVER_STREAMING: {
    id: 'test-tt-inference-server-streaming',
    name: 'Test tt-inference-server (Streaming)',
    template: 'curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \'{"model": "Llama-3.1-8B-Instruct", "prompt": "Write a haiku about AI acceleration:", "max_tokens": 100, "stream": true}\'',
    description: 'Tests streaming responses from vLLM server (Server-Sent Events)',
  },

  TEST_TT_INFERENCE_SERVER_SAMPLING: {
    id: 'test-tt-inference-server-sampling',
    name: 'Test tt-inference-server (Sampling)',
    template: 'echo "=== High Temperature (Creative) ===" && curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \'{"model": "Llama-3.1-8B-Instruct", "prompt": "Once upon a time", "max_tokens": 50, "temperature": 1.2, "top_p": 0.95}\' && echo "\n\n=== Low Temperature (Deterministic) ===" && curl -X POST http://localhost:8000/v1/completions -H "Content-Type: application/json" -d \'{"model": "Llama-3.1-8B-Instruct", "prompt": "The capital of France is", "max_tokens": 10, "temperature": 0.1}\'',
    description: 'Tests different sampling parameters with the OpenAI-compatible API',
  },

  CREATE_TT_INFERENCE_SERVER_CLIENT: {
    id: 'create-tt-inference-server-client',
    name: 'Create Python Client for tt-inference-server',
    template: 'cat > ~/tt-scratchpad/tt-inference-client.py << \'EOF\'\nfrom openai import OpenAI\n\n# Point to the vLLM server started by tt-inference-server\nclient = OpenAI(\n    base_url="http://localhost:8000/v1",\n    api_key="dummy"  # Not used, but required by SDK\n)\n\ndef query_inference_server(prompt, max_tokens=100, temperature=0.7):\n    """Query vLLM server using OpenAI SDK"""\n    try:\n        response = client.completions.create(\n            model="Llama-3.1-8B-Instruct",\n            prompt=prompt,\n            max_tokens=max_tokens,\n            temperature=temperature\n        )\n        \n        generated_text = response.choices[0].text\n        print(f"Generated text: {generated_text}")\n        print(f"Tokens: {response.usage.total_tokens}")\n        return response\n    except Exception as e:\n        print(f"Error: {e}")\n        return None\n\nif __name__ == "__main__":\n    query_inference_server(\n        "Explain quantum computing to a 5-year-old:",\n        max_tokens=100,\n        temperature=0.8\n    )\nEOF\nchmod +x ~/tt-scratchpad/tt-inference-client.py && echo "✓ Created ~/tt-scratchpad/tt-inference-client.py (uses OpenAI SDK)"',
    description: 'Creates a Python client using OpenAI SDK to connect to the vLLM server',
  },

  CREATE_TT_INFERENCE_SERVER_CONFIG: {
    id: 'create-tt-inference-server-config',
    name: 'Create tt-inference-server Config File',
    template: 'echo "⚠️  Note: tt-inference-server does not use config files. Use command-line arguments instead:" && echo "Example: python3 run.py --model Llama-3.1-8B-Instruct --device n150 --workflow server --docker-server"',
    description: 'Shows that tt-inference-server uses command-line arguments, not config files',
  },

  // ========================================
  // Lesson 8: Image Generation (Stable Diffusion 3.5 Large)
  // ========================================

  // Image Generation (Lesson 8) - Stable Diffusion 3.5 Large
  GENERATE_RETRO_IMAGE: {
    id: 'generate-retro-image',
    name: 'Generate Sample Image with SD 3.5',
    template:
      'mkdir -p ~/tt-scratchpad && cd ~/tt-scratchpad && export PYTHONPATH="{{ttMetalPath}}":$PYTHONPATH && export MESH_DEVICE=N150 && export NO_PROMPT=1 && pytest "{{ttMetalPath}}"/models/experimental/stable_diffusion_35_large/demo.py',
    description: 'Generates a sample 1024x1024 image using Stable Diffusion 3.5 Large on TT hardware, saves to ~/tt-scratchpad',
    variables: ['ttMetalPath'],
  },

  START_INTERACTIVE_IMAGE_GEN: {
    id: 'start-interactive-image-gen',
    name: 'Start Interactive SD 3.5 Mode',
    template:
      'mkdir -p ~/tt-scratchpad && cd ~/tt-scratchpad && export PYTHONPATH="{{ttMetalPath}}":$PYTHONPATH && export MESH_DEVICE=N150 && export NO_PROMPT=0 && pytest "{{ttMetalPath}}"/models/experimental/stable_diffusion_35_large/demo.py',
    description: 'Starts interactive mode where you can enter custom prompts for image generation, saves to ~/tt-scratchpad',
    variables: ['ttMetalPath'],
  },

  COPY_IMAGE_GEN_DEMO: {
    id: 'copy-image-gen-demo',
    name: 'Copy SD 3.5 Demo to Scratchpad',
    template:
      'mkdir -p ~/tt-scratchpad && cp "{{ttMetalPath}}"/models/experimental/stable_diffusion_35_large/demo.py ~/tt-scratchpad/sd35_demo.py && echo "✓ Demo copied to ~/tt-scratchpad/sd35_demo.py - ready to edit!"',
    description: 'Copies the Stable Diffusion demo to ~/tt-scratchpad for experimentation and modification',
    variables: ['ttMetalPath'],
  },




  // Image Classification with TT-Forge (Lesson 11)
  // forge + torch-xla live in ~/tt-forge-venv (symlink → /opt/venv-forge in developer images).
  BUILD_FORGE_FROM_SOURCE: {
    id: 'build-forge-from-source',
    name: 'Activate Forge Environment',
    template: `source ~/tt-forge-venv/bin/activate && python3 -c "\n${FORGE_ACTIVATION_PY}\n"`,
    description: 'Activates the pre-installed venv-forge environment and verifies the forge stack',
  },

  INSTALL_FORGE: {
    id: 'install-forge',
    name: 'Activate Forge Environment',
    template: `source ~/tt-forge-venv/bin/activate && python3 -c "\n${FORGE_ACTIVATION_PY}\n"`,
    description: 'Activates the pre-installed venv-forge environment (forge is pre-installed, no pip needed)',
  },

  TEST_FORGE_INSTALL: {
    id: 'test-forge-install',
    name: 'Verify Forge Stack',
    template: `source ~/tt-forge-venv/bin/activate && python3 -c "\n${FORGE_FULL_VERIFY_PY}\n"`,
    description: 'Imports forge, jax, and torch_xla and prints their versions + visible TT devices',
  },



  RUN_FORGE_CLASSIFIER: {
    id: 'run-forge-classifier',
    name: 'Run Image Classifier',
    template: 'source ~/tt-forge-venv/bin/activate && cd ~/tt-scratchpad && python3 tt-forge-classifier.py',
    description: 'Activates venv-forge and runs MobileNetV2 image classification on TT hardware',
  },



  // ──────────────────────────────────────────────────────────────────────────
  // venv-forge activation & verification (shared by forge + XLA lessons)
  // ──────────────────────────────────────────────────────────────────────────

  ACTIVATE_FORGE_ENV: {
    id: 'activate-forge-env',
    name: 'Activate Forge Environment',
    template: `source ~/tt-forge-venv/bin/activate && python3 -c "\n${JAX_DEVICE_CHECK_PY}\n"`,
    description: 'Activates the pre-installed venv-forge environment and prints visible TT devices',
  },

  VERIFY_FORGE_STACK: {
    id: 'verify-forge-stack',
    name: 'Verify Forge Stack',
    template: `source ~/tt-forge-venv/bin/activate && python3 -c "\n${FORGE_FULL_VERIFY_PY}\n"`,
    description: 'Imports forge, jax, and torch_xla and prints their versions + visible TT devices',
  },

  // TT-XLA JAX Integration
  RUN_JAX_QUICKSTART: {
    id: 'run-jax-quickstart',
    name: 'Run JAX Quickstart',
    template: "source ~/tt-forge-venv/bin/activate && python3 -c \"\ntry:\n    import tt_torch  # pre-loads TT PJRT plugin so jax.devices() sees TT hardware\nexcept ImportError:\n    pass\nimport jax\nimport jax.numpy as jnp\na = jnp.ones((1024, 1024))\nb = jnp.ones((1024, 1024))\nc = a @ b\nprint('shape  :', c.shape)\nprint('device :', c.devices())\nprint('c[0,0] :', c[0, 0])  # expect 1024.0\n\"",
    description: 'Runs a 1024x1024 JAX matmul on TT hardware via venv-forge; prints shape, device, value',
  },

  RUN_JAX_PMAP_DEMO: {
    id: 'run-jax-pmap-demo',
    name: 'Run JAX pmap Demo (multi-device)',
    template: "source ~/tt-forge-venv/bin/activate && python3 -c \"\ntry:\n    import tt_torch  # pre-loads TT PJRT plugin so jax.devices() sees TT hardware\nexcept ImportError:\n    pass\nimport jax\nimport jax.numpy as jnp\ndevices = jax.devices()\nn = len(devices)\nprint(f'Running across {n} TT device(s)')\n@jax.pmap\ndef matmul_per_device(A):\n    return A @ A.T\nA = jnp.ones((n, 512, 512))\nresult = matmul_per_device(A)\nprint('result shape :', result.shape)\nprint('sharding     :', result.sharding)\n\"",
    description: 'Maps a matmul across all TT devices using jax.pmap (QB2 uses all 4 chips)',
  },

  RUN_PYTORCH_XLA_DEMO: {
    id: 'run-pytorch-xla-demo',
    name: 'Run PyTorch/XLA Demo',
    template: "source ~/tt-forge-venv/bin/activate && python3 -c \"\nimport torch\nimport torch_xla.core.xla_model as xm\ndevice = xm.xla_device()\nprint('TT device:', device)\nx = torch.randn(256, 256).to(device)\ny = torch.randn(256, 256).to(device)\nz = x @ y\nxm.mark_step()\nprint('z.shape :', z.shape)\nprint('z.device:', z.device)\n\"",
    description: 'Runs a PyTorch matmul on TT hardware via torch-xla (both pre-installed in venv-forge)',
  },

  DOWNLOAD_TT_XLA_DEMO: {
    id: 'download-tt-xla-demo',
    name: 'Download TT-XLA GPT-2 Demo',
    template: 'git clone https://github.com/tenstorrent/tt-forge.git ~/tt-forge 2>/dev/null || (cd ~/tt-forge && git pull origin main)',
    description: 'Clones the tt-forge repo (contains GPT-2, ALBERT, OPT, ResNet JAX demos)',
  },

  RUN_TT_XLA_DEMO: {
    id: 'run-tt-xla-demo',
    name: 'Run TT-XLA GPT-2 Demo',
    template: 'source ~/tt-forge-venv/bin/activate && cd ~/tt-forge/demos/tt-xla/nlp/jax && pip install -q -r requirements.txt && python3 gpt_demo.py',
    description: 'Runs official GPT-2 next-token demo via JAX on TT hardware',
  },

  // These commands keep the registered handlers for installTtXla and testTtXlaInstall
  // wired up (they display a helpful redirect rather than doing nothing).
  INSTALL_TT_XLA: {
    id: 'install-tt-xla',
    name: 'Activate Forge Environment',
    template: `source ~/tt-forge-venv/bin/activate && python3 -c "\n${JAX_DEVICE_CHECK_PY}\n"`,
    description: 'venv-forge is pre-installed — activates the environment instead',
  },

  CREATE_TT_XLA_TEST: {
    id: 'create-tt-xla-test',
    name: 'Run JAX Quickstart',
    template: "source ~/tt-forge-venv/bin/activate && python3 -c \"\ntry:\n    import tt_torch  # pre-loads TT PJRT plugin so jax.devices() sees TT hardware\nexcept ImportError:\n    pass\nimport jax, jax.numpy as jnp\nc = jnp.ones((1024, 1024)) @ jnp.ones((1024, 1024))\nprint('shape:', c.shape, 'device:', c.devices())\n\"",
    description: 'Runs a quick JAX matmul to verify TT-XLA is working',
  },

  TEST_TT_XLA_INSTALL: {
    id: 'test-tt-xla-install',
    name: 'Verify Forge Stack',
    template: "source ~/tt-forge-venv/bin/activate && python3 -c \"\ntry:\n    import tt_torch  # pre-loads TT libs so JAX PJRT plugin can find them\nexcept ImportError:\n    pass\nimport jax\nprint(jax.devices())\n\"",
    description: 'Verifies TT-XLA is accessible in venv-forge',
  },

  // ========================================
  // Lesson 13: RISC-V Programming on Tensix Cores
  // ========================================

  BUILD_PROGRAMMING_EXAMPLES: {
    id: 'build-programming-examples',
    name: 'Build Programming Examples',
    template: 'cd ~/tt-metal && ./build_metal.sh --build-programming-examples',
    description: 'Builds tt-metal with programming examples including RISC-V demonstrations',
  },

  RUN_RISCV_EXAMPLE: {
    id: 'run-riscv-example',
    name: 'Run RISC-V Addition Example',
    template: 'cd ~/tt-metal && export TT_METAL_DPRINT_CORES=0,0 && ./build_Release/programming_examples/metal_example_add_2_integers_in_riscv',
    description: 'Runs the RISC-V addition example on BRISC processor',
  },

  // ========================================
  // Lesson 15: TT-Metalium Cookbook
  // ========================================

  RUN_GAME_OF_LIFE: {
    id: 'run-game-of-life',
    name: 'Run Game of Life',
    template: 'cd ~/tt-scratchpad/cookbook/game_of_life && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 game_of_life.py',
    description: 'Runs Conway\'s Game of Life with random initial state',
  },

  RUN_GAME_OF_LIFE_GLIDER: {
    id: 'run-game-of-life-glider',
    name: 'Run Game of Life (Glider)',
    template: 'cd ~/tt-scratchpad/cookbook/game_of_life && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 -c "from game_of_life import GameOfLife; from visualizer import animate_game_of_life; import ttnn; device = ttnn.open_device(device_id=0); game = GameOfLife(device, grid_size=(256, 256)); initial = game.initialize_pattern(\'glider\'); history = game.simulate(initial, num_generations=200); animate_game_of_life(history, interval=50); ttnn.close_device(device)"',
    description: 'Runs Game of Life with classic glider pattern',
  },

  RUN_GAME_OF_LIFE_GLIDER_GUN: {
    id: 'run-game-of-life-glider-gun',
    name: 'Run Game of Life (Glider Gun)',
    template: 'cd ~/tt-scratchpad/cookbook/game_of_life && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 -c "from game_of_life import GameOfLife; from visualizer import animate_game_of_life; import ttnn; device = ttnn.open_device(device_id=0); game = GameOfLife(device, grid_size=(256, 256)); initial = game.initialize_pattern(\'glider_gun\'); history = game.simulate(initial, num_generations=500); animate_game_of_life(history, interval=50); ttnn.close_device(device)"',
    description: 'Runs Game of Life with Gosper Glider Gun (generates gliders infinitely)',
  },

  RUN_MANDELBROT_EXPLORER: {
    id: 'run-mandelbrot-explorer',
    name: 'Run Mandelbrot Explorer',
    template: 'cd ~/tt-scratchpad/cookbook/mandelbrot && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 -c "from renderer import MandelbrotRenderer; from explorer import MandelbrotVisualizer; import ttnn; device = ttnn.open_device(device_id=0); renderer = MandelbrotRenderer(device); viz = MandelbrotVisualizer(renderer); viz.interactive_explorer(width=1024, height=1024); ttnn.close_device(device)"',
    description: 'Launches interactive Mandelbrot explorer with click-to-zoom',
  },

  RUN_MANDELBROT_JULIA: {
    id: 'run-mandelbrot-julia',
    name: 'Run Julia Sets Comparison',
    template: 'cd ~/tt-scratchpad/cookbook/mandelbrot && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 -c "from renderer import MandelbrotRenderer; from explorer import MandelbrotVisualizer; import ttnn; device = ttnn.open_device(device_id=0); renderer = MandelbrotRenderer(device); viz = MandelbrotVisualizer(renderer); c_values = [(-0.4, 0.6), (0.285, 0.01), (-0.70176, -0.3842), (-0.835, -0.2321), (-0.8, 0.156), (0.0, -0.8)]; viz.compare_julia_sets(c_values); ttnn.close_device(device)"',
    description: 'Display 6 interesting Julia set fractals side-by-side',
  },

  RUN_AUDIO_PROCESSOR: {
    id: 'run-audio-processor',
    name: 'Run Audio Processor Demo',
    template: 'cd ~/tt-scratchpad/cookbook/audio_processor && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 processor.py',
    description: 'Runs audio processor with mel-spectrogram visualization',
  },

  RUN_IMAGE_FILTERS: {
    id: 'run-image-filters',
    name: 'Run Image Filters Demo',
    template: 'cd ~/tt-scratchpad/cookbook/image_filters && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 filters.py',
    description: 'Runs image filter examples (edge detect, blur, sharpen, emboss)',
  },

  RUN_PARTICLE_LIFE: {
    id: 'run-particle-life',
    name: 'Run Particle Life Simulation',
    template: 'cd ~/tt-scratchpad/cookbook/particle_life && export PYTHONPATH=~/tt-metal:$PYTHONPATH && python3 test_particle_life.py',
    description: 'Runs Particle Life simulation with emergent complexity patterns',
  },

  // ========================================
  // Lesson 17: Native Video Animation with AnimateDiff
  // ========================================


  RUN_ANIMATEDIFF_2FRAME: {
    id: 'run-animatediff-phase1',
    name: 'Run Phase 1 — CPU AnimateDiffPipeline',
    template: 'cd ~/tt-projects/tt-animatediff && python3 examples/generate.py --mode cpu --prompt "purple phosphor glow across distant mountains at 2am, retro CRT haze, cyan mist, cinematic" --frames 8 --steps 25 --output output/phase1.gif 2>&1 | grep -v "DEBUG\\|Config{"',
    description: 'Generate animated frames on CPU using diffusers AnimateDiffPipeline with MotionAdapter',
  },

  RUN_ANIMATEDIFF_16FRAME: {
    id: 'run-animatediff-phase2',
    name: 'Run Phase 2.5 — Blackhole TTNN UNet + temporal attention',
    template: `cd ~/tt-metal && source python_env/bin/activate && cd ~/tt-projects/tt-animatediff && python3 examples/generate.py --mode blackhole --prompt "1939 World's Fair imagined from the year 2099, art deco spires at golden dusk, retro-futurist optimism, cinematic 4K" --frames 8 --steps 25 --temporal-alpha 0.35 --output output/blackhole.gif 2>&1 | grep -v "DEBUG\\|Config{" | grep -v "^2026\\|UMD"`,
    description: 'Generate frames using TTNN UNet + cross-frame temporal attention on Blackhole hardware (P100/P300C/QB2)',
  },


  SETUP_ANIMATEDIFF_PROJECT: {
    id: 'setup-animatediff-project',
    name: 'Setup AnimateDiff Project',
    template: 'mkdir -p ~/tt-projects && git clone --depth 1 --branch v0.1.0 https://github.com/tenstorrent/tt-animatediff.git ~/tt-projects/tt-animatediff 2>&1 || (cd ~/tt-projects/tt-animatediff && git fetch --tags && git checkout v0.1.0) && cd ~/tt-projects/tt-animatediff && python3 -m pip install -e ".[dev]" && python3 -c "import animatediff_ttnn; print(\'✓ tt-animatediff v0.1.0 ready at ~/tt-projects/tt-animatediff/\')"',
    description: 'Clones tt-animatediff v0.1.0 from GitHub into ~/tt-projects/tt-animatediff and installs it',
  },

  // ========================================
  // Custom Training Lessons
  // ========================================

  // Verified build recipe: content/templates/llm-from-scratch/BUILD_TTML.md
  // ttml (tt-train) is source-only — there is no pip wheel and tt-train has
  // no pyproject.toml, so `pip install -e .` (the old approach) always
  // failed. The real recipe configures tt-train as a tt-metal cmake
  // subproject, builds the `_ttml` bindings, then REBUILDS ttnn's
  // `_ttnn.so` and copies it over the prebuilt one. That rebuild is
  // required: every prebuilt tt-metal image (including TT-QuietBox 2)
  // ships a `_ttnn.so` with a different nanobind ABI tag than a freshly
  // built `_ttml`, and skipping the rebuild is the #1 cause of
  // `std::bad_cast` on `import ttml`. Finally, since there's no install
  // step to wire ttml onto the venv, we add a `.pth` file pointing at the
  // built sources.
  INSTALL_TT_TRAIN: {
    id: 'install-tt-train',
    name: 'Install tt-train',
    template:
      'export TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tt-metal}" && ' +
      'export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME" && ' +
      ': "${TT_METAL_ARCH_NAME:=wormhole_b0}" && export TT_METAL_ARCH_NAME && ' +
      'export CMAKE_POLICY_VERSION_MINIMUM=3.5 && ' +
      'export TT_LOGGER_LEVEL=FATAL && ' +
      'cd "$TT_METAL_HOME" && ' +
      'echo "Configuring tt-train subproject..." && ' +
      './build_metal.sh --build-tt-train --configure-only && ' +
      'echo "Building ttml python bindings (a few minutes with warm ccache)..." && ' +
      'cmake --build build_Release --target _ttml && ' +
      'echo "Rebuilding ttnn _ttnn.so so its ABI matches ttml (fixes std::bad_cast on import ttml)..." && ' +
      'ninja -C build_Release ttnn/_ttnn.so && ' +
      'cp -a build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so && ' +
      'SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\'purelib\'])") && ' +
      '{ echo "$TT_METAL_HOME/tt-train/sources/ttml"; echo "$TT_METAL_HOME/build_Release/tt-train/sources/ttml"; echo "$TT_METAL_HOME/build/tt-train/sources/ttml"; } > "$SITE_PACKAGES/ttml-custom.pth" && ' +
      'python3 -c "import ttml, ttnn; print(\'\\u2713 ttml + ttnn installed and importable\')"',
    description:
      'Builds ttml (tt-train) from source against $TT_METAL_HOME: configures the tt-train subproject, builds the _ttml bindings, rebuilds ttnn/_ttnn.so to fix the nanobind ABI mismatch (std::bad_cast), and wires ttml onto the active venv via a .pth file',
  },

  // CT-8: Training from Scratch Commands
  PREPARE_SHAKESPEARE: {
    id: 'prepare-shakespeare',
    name: 'Prepare Shakespeare Dataset',
    template: 'cd ~/tt-scratchpad/training/data && python prepare_shakespeare.py --output shakespeare.txt --split',
    description: 'Download and prepare tiny-shakespeare dataset for training from scratch',
  },


  TRAIN_FROM_SCRATCH: {
    id: 'train-from-scratch',
    name: 'Train from Scratch',
    template: 'cd ~/tt-scratchpad/training && python train_from_scratch.py --config configs/nano_trickster.yaml',
    description: 'Train nano-trickster from random initialization on tiny-shakespeare',
  },


  // ========================================
  // QB2 Video Generation (qb2-video-generation)
  // ========================================

  CLONE_TT_LOCAL_GENERATOR: {
    id: 'clone-tt-local-generator',
    name: 'Clone tt-local-generator',
    template: 'git clone https://github.com/tenstorrent/tt-local-generator.git ~/code/tt-local-generator',
    description: 'Clones the tt-local-generator video generation UI to ~/code/tt-local-generator',
  },

  SETUP_VIDEO_GEN_VENDOR: {
    id: 'setup-video-gen-vendor',
    name: 'Set Up Vendored Inference Server',
    template: 'cd ~/code/tt-local-generator && ./bin/setup_vendor.sh',
    description: 'Clones the pinned tt-inference-server commit into vendor/ (shallow clone, never touches ~/code/tt-inference-server)',
  },

  APPLY_VIDEO_GEN_PATCHES: {
    id: 'apply-video-gen-patches',
    name: 'Apply QB2 Hotpatches',
    template: 'cd ~/code/tt-local-generator && ./bin/apply_patches.sh',
    description: 'Applies QB2 hotpatches (config overrides, pipeline fixes) to the vendored tt-inference-server',
  },

  DOWNLOAD_WAN22_MODEL: {
    id: 'download-wan22-model',
    name: 'Download Wan2.2-T2V-A14B Model',
    template: 'hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers',
    description: 'Downloads the 14B Wan2.2 text-to-video model from HuggingFace to HF cache (~118 GB, one-time)',
  },

  DOWNLOAD_QWEN3_SMALL: {
    id: 'download-qwen3-small',
    name: 'Download Qwen3-0.6B Prompt Model',
    template: 'mkdir -p ~/models && hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B',
    description: 'Downloads Qwen3-0.6B to ~/models/Qwen3-0.6B (~1.2 GB); ~/models can be a symlink to any storage location',
  },

  START_WAN22_SERVER: {
    id: 'start-wan22-server',
    name: 'Start Wan2.2 Video Server',
    template: 'cd ~/code/tt-local-generator && ./bin/start_wan_qb2.sh',
    description: 'Starts Wan2.2-T2V-A14B-Diffusers on QB2 (P300X2). First-run compilation takes ~9 min; subsequent starts take ~5 min.',
  },

  START_PROMPT_GEN_SERVER: {
    id: 'start-prompt-gen-server',
    name: 'Start Prompt Generation Server',
    template: 'cd ~/code/tt-local-generator && ./bin/start_prompt_gen.sh',
    description: 'Starts the Qwen3-0.6B prompt polish server on port 8001 (CPU-only, ~2.9 GB RAM)',
  },

  CHECK_VIDEO_SERVER_HEALTH: {
    id: 'check-video-server-health',
    name: 'Check Server Health',
    template: 'echo "=== Video server (port 8000) ===" && curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health && echo "\n=== Prompt server (port 8001) ===" && curl -s http://localhost:8001/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8001/health',
    description: 'Checks health of both the video inference server (port 8000) and prompt server (port 8001)',
  },

  LAUNCH_TT_GEN: {
    id: 'launch-tt-gen',
    name: 'Launch tt-gen GUI',
    template: 'cd ~/code/tt-local-generator && ./tt-gen',
    description: 'Launches the GTK4 video generation GUI. Requires python3-gi (apt package, not venv).',
  },

  // ========================================
  // QB2 Local Agents (qb2-local-agents lesson)
  // ========================================



  CHECK_AGENT_SERVER_HEALTH: {
    id: 'check-agent-server-health',
    name: 'Check Agent Server Health',
    template: 'curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null || curl http://localhost:8000/v1/models',
    description: 'Checks the vLLM server on port 8000 and lists available models',
  },

  CLONE_TT_AGENTS: {
    id: 'clone-tt-agents',
    name: 'Clone tt-agents Repository',
    // tt-agents currently lives under the tsingletaryTT org, not tenstorrent — it hasn't
    // moved into the Tenstorrent org yet.
    //
    // Two things this template is careful about:
    //
    //  1. It branches on whether a git repo already exists rather than running the clone and
    //     discarding stderr. The old `git clone ... 2>/dev/null || (git pull)` form hid every
    //     clone failure — bad URL, DNS, auth, unwritable ~/code — behind a `git pull` against
    //     a directory that may not exist, so real errors surfaced as confusing pull errors.
    //
    //  2. The update path rewrites `origin` before pulling. The repo moved orgs, so anyone who
    //     ran an earlier version of this button has origin pointing at tenstorrent/tt-agents.
    //     Without the rewrite they would pull from the old org indefinitely and never receive
    //     tt_agents_common.py, the shared helper the framework demos (01-05) import.
    template: `mkdir -p ~/code
if [ -d ~/code/tt-agents/.git ]; then
  echo "Updating existing clone at ~/code/tt-agents"
  cd ~/code/tt-agents && git remote set-url origin https://github.com/tsingletaryTT/tt-agents.git && git pull origin main
else
  git clone https://github.com/tsingletaryTT/tt-agents.git ~/code/tt-agents && cd ~/code/tt-agents
fi && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt || echo "⚠️  tt-agents setup did not complete — see the errors above"`,
    description: 'Clones (or updates, re-pointing origin at the tsingletaryTT org) tt-agents to ~/code/tt-agents and installs Python dependencies',
  },

  COPY_AGENTS_TO_SCRATCHPAD: {
    id: 'copy-agents-to-scratchpad',
    name: 'Copy Agent Scripts to Scratchpad',
    template: 'mkdir -p ~/tt-scratchpad/agents && cp ~/code/tt-agents/*.py ~/tt-scratchpad/agents/ && cp ~/code/tt-agents/requirements.txt ~/tt-scratchpad/agents/ && cp ~/code/tt-agents/world.json ~/tt-scratchpad/agents/ && echo "✓ Agent scripts copied to ~/tt-scratchpad/agents/"',
    description: 'Copies all agent scripts, requirements, and world.json to ~/tt-scratchpad/agents/ for safe experimentation',
  },

  RUN_AGENTS_VERIFY: {
    id: 'run-agents-verify',
    name: 'Run Agent Tool Verification',
    template: 'python3 ~/code/tt-agents/00_verify_tools.py',
    description: 'Runs the tool-calling verification script: confirms inference, tool calls, and structured output all work',
  },

  RUN_RESEARCH_AGENT: {
    id: 'run-research-agent',
    name: 'Run Research Agent (Demo 1)',
    template: 'python3 ~/code/tt-agents/01_research_agent.py',
    description: 'Runs Demo 1: smolagents CodeAgent that searches the web and synthesizes a cited report',
  },

  RUN_CODE_EXPLORER: {
    id: 'run-code-explorer',
    name: 'Run Code Explorer (Demo 2)',
    template: 'python3 ~/code/tt-agents/02_code_explorer.py',
    description: 'Runs Demo 2: OpenAI Agents SDK tool-calling agent that explores a local codebase',
  },

  RUN_WRITING_PIPELINE: {
    id: 'run-writing-pipeline',
    name: 'Run Writing Pipeline (Demo 3)',
    template: 'python3 ~/code/tt-agents/03_writing_pipeline.py',
    description: 'Runs Demo 3: CrewAI multi-agent writing pipeline (Researcher → Writer → Editor) with format picker',
  },

  RUN_DUNGEON_MASTER: {
    id: 'run-dungeon-master',
    name: 'Run Dungeon Master (Demo 4)',
    template: 'python3 ~/code/tt-agents/04_dungeon_master.py',
    description: 'Runs Demo 4: smolagents ToolCallingAgent persistent-world dungeon master with generative tools',
  },

  RUN_STORYBOARD_PIPELINE: {
    id: 'run-storyboard-pipeline',
    name: 'Run Storyboard Pipeline (Demo 5)',
    template: 'python3 ~/code/tt-agents/05_storyboard_to_pixelart.py --simulate',
    description: 'Runs Demo 5: two-stage CrewAI + smolagents pipeline demonstrating model lifecycle — storyboard (70B) → pixel art prompts (32B)',
  },
  RUN_LANDSCAPE_SVG: {
    id: 'run-landscape-svg',
    name: 'Generate Landscape SVG (Demo 6)',
    template: 'python3 ~/code/tt-agents/06_landscape_svg.py --simulate',
    description: 'Runs Demo 6: parameterized generative landscape SVG — direct LLM → SVG with gradients, terrain, clouds, stars',
  },

  // ========================================
  // ttsim: Twenty-and-Ten Lesson
  // ========================================

  SETUP_TTSIM: {
    id: 'setup-ttsim',
    name: 'Set Up ttsim Simulator',
    // Bump TTSIM_VERSION in one place to move the lesson to a newer release.
    // Keep it in sync with the Setup block in content/lessons/ttsim-twenty-and-ten.md.
    // Config naming: the _xN suffix is the chip count for a whole board —
    //   wh_x2 = N300, bh_x2 = P300, wh_x8 = T3000, bh_x4 = TT-QuietBox 2 (2x P300).
    // libttsim_qsr.so is *Quasar*, a separate pre-silicon architecture, and is
    // deliberately not downloaded here.
    template: `mkdir -p ~/sim
TTSIM_VERSION=v1.10.1
for SIM in wh bh wh_x2 bh_x2 wh_x8 bh_x4; do
  wget -q https://github.com/tenstorrent/ttsim/releases/download/$TTSIM_VERSION/libttsim_$SIM.so -O ~/sim/libttsim_$SIM.so || { echo "ERROR: failed to download libttsim_$SIM.so"; exit 1; }
done
if [ -n "$TT_METAL_HOME" ]; then
  cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml || { echo "ERROR: failed to copy SOC descriptor"; exit 1; }
  cp $TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/n300_cluster_desc.yaml ~/sim/n300_cluster_desc.yaml || { echo "WARNING: n300 cluster desc copy skipped (optional for N300 sim)"; }
else
  echo "TT_METAL_HOME not set — SOC descriptor copy skipped"
fi
echo "ttsim $TTSIM_VERSION ready (wh + bh + wh_x2 + bh_x2 + wh_x8 + bh_x4)"`,
    description: 'Downloads the pinned ttsim Wormhole, Blackhole, and multi-chip (wh_x2/bh_x2/wh_x8/bh_x4) binaries and copies SOC descriptors',
  },

  RUN_TTSIM_ATTENTION: {
    id: 'run-ttsim-attention',
    name: 'Run Transformer Attention on ttsim',
    template: `export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
python3 ~/tt-scratchpad/ttsim/ttsim_attention.py`,
    description: 'Runs a transformer attention layer forward pass on the ttsim Wormhole simulator',
  },

  // ========================================
  // ttsim QEMU Bridge
  // ========================================

  // Boot command for ttsim-qemu VM.
  // Requires qemu-system-x86_64 built from tenstorrent/ttsim-qemu fork — the
  // fork adds the -device ttsim option; -netdev user is standard upstream QEMU.
  // Template vars:
  //   {{IMAGE_PATH}} — path to .qcow2 VM image
  //   {{LIB_PATH}}   — path to libttsim_wh.so or libttsim_bh.so
  //   {{PID_FILE}}   — path for the QEMU PID file
  BOOT_TTSIM_QEMU: {
    id: 'boot-ttsim-qemu',
    name: 'Boot ttsim QEMU Bridge',
    // -cpu max: required in TCG mode to expose full x86 feature set (vmovups etc.)
    // bar4-size: Wormhole = 32M, Blackhole = 32G — filled via {{BAR4_SIZE}} at boot time
    // No -enable-kvm: KVM MMIO emulator can't handle 16-byte WC-mapped TLB reads
    template: `qemu-system-x86_64 \\\n  -m 8G -smp 4 \\\n  -cpu max \\\n  -drive file="{{IMAGE_PATH}}",if=virtio,snapshot=on \\\n  -device ttsim,lib="{{LIB_PATH}}",bar4-size={{BAR4_SIZE}} \\\n  -netdev user,id=net0,hostfwd=tcp::2222-:22 \\\n  -device virtio-net-pci,netdev=net0 \\\n  -serial file:/tmp/ttsim-qemu-serial.log \\\n  -chardev socket,id=mon,path=/tmp/ttsim-mon.sock,server=on,wait=off \\\n  -mon chardev=mon,mode=readline \\\n  -display none \\\n  -daemonize \\\n  -pidfile "{{PID_FILE}}"`,
    description: 'Boots an Ubuntu VM with the ttsim PCI device attached (WH: bar4-size=32M, BH: bar4-size=32G)',
  },

  SSH_TTSIM_QEMU: {
    id: 'ssh-ttsim-qemu',
    name: 'SSH into ttsim QEMU Bridge',
    // UserKnownHostsFile=/dev/null: snapshot=on regenerates host keys on each boot;
    // without this, known_hosts mismatches cause "REMOTE HOST IDENTIFICATION HAS CHANGED"
    // even with StrictHostKeyChecking=no.
    template: `ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=30 ubuntu@localhost`,
    description: 'Opens an SSH terminal session inside the running ttsim QEMU Bridge VM',
  },
};

/**
 * Replaces variables in a command template with actual values
 *
 * @param template - Command template with {{variable}} placeholders
 * @param variables - Object with variable names and their values
 * @returns Command string with variables replaced
 */
export function replaceVariables(template: string, variables: Record<string, string>): string {
  let result = template;

  for (const [key, value] of Object.entries(variables)) {
    const placeholder = `{{${key}}}`;
    result = result.replace(new RegExp(placeholder, 'g'), value);
  }

  return result;
}

/**
 * Gets a formatted command for display in markdown
 *
 * @param commandKey - Key from TERMINAL_COMMANDS
 * @param variables - Variables to replace (if any)
 * @returns Formatted command string ready for display
 */
export function getDisplayCommand(
  commandKey: keyof typeof TERMINAL_COMMANDS,
  variables?: Record<string, string>
): string {
  const command = TERMINAL_COMMANDS[commandKey];

  if (variables && command.variables) {
    return replaceVariables(command.template, variables);
  }

  return command.template;
}
