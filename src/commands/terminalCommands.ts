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

import { MODEL_REGISTRY, DEFAULT_MODEL_KEY, type ModelConfig } from '../config';

/**
 * Get the default model config
 */
function getDefaultModel(): ModelConfig {
  return MODEL_REGISTRY[DEFAULT_MODEL_KEY];
}

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

/**
 * All terminal commands used in the walkthrough
 */
export const TERMINAL_COMMANDS: Record<string, CommandTemplate> = {
  // ========================================
  // Lesson 0: Modern Setup with tt-installer 2.0
  // ========================================

  QUICK_INSTALL: {
    id: 'quick-install',
    name: 'Quick Install with tt-installer',
    template: '/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"',
    description: 'One-command installation of the full Tenstorrent stack (interactive prompts)',
  },

  DOWNLOAD_INSTALLER: {
    id: 'download-installer',
    name: 'Download tt-installer Script',
    template: 'cd ~ && curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh -O && chmod +x install.sh',
    description: 'Downloads the tt-installer script for inspection and customization',
  },

  RUN_INTERACTIVE_INSTALL: {
    id: 'run-interactive-install',
    name: 'Run Interactive Installation',
    template: 'cd ~ && ./install.sh',
    description: 'Runs tt-installer with interactive prompts for customization',
  },

  RUN_NON_INTERACTIVE_INSTALL: {
    id: 'run-non-interactive-install',
    name: 'Run Non-Interactive Installation',
    template: 'cd ~ && ./install.sh --mode-non-interactive --python-choice=new-venv --install-metalium-models-container=off --reboot-option=never',
    description: 'Runs tt-installer in automated mode with recommended defaults',
  },

  TEST_METALIUM_CONTAINER: {
    id: 'test-metalium-container',
    name: 'Test tt-metalium Container',
    template: 'tt-metalium "python3 -c \'import ttnn; print(f\\"TTNN version: {ttnn.__version__}\\"); print(\\"✅ tt-metalium container working!\\")\'"',
    description: 'Verifies tt-metalium container is installed and TTNN is accessible',
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

  INSTALL_DEPENDENCIES: {
    id: 'install-dependencies',
    name: 'Install System Dependencies',
    template: 'cd ~/tt-metal && sudo ./install_dependencies.sh',
    description: 'Installs required system libraries, drivers, and dependencies for tt-metal',
  },

  COPY_ENVIRONMENT_SETUP: {
    id: 'copy-environment-setup',
    name: 'Copy Environment Setup Commands',
    template: 'export TT_METAL_HOME=~/tt-metal\nexport PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH\nexport LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH',
    description: 'Sets up environment variables for tt-metal (TT_METAL_HOME, PYTHONPATH, LD_LIBRARY_PATH)',
  },

  PERSIST_ENVIRONMENT: {
    id: 'persist-environment',
    name: 'Add Environment Variables to ~/.bashrc',
    template: '(grep -q "export TT_METAL_HOME=~/tt-metal" ~/.bashrc || echo \'export TT_METAL_HOME=~/tt-metal\' >> ~/.bashrc) && (grep -q "export PYTHONPATH=\\$TT_METAL_HOME:\\$PYTHONPATH" ~/.bashrc || echo \'export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH\' >> ~/.bashrc) && (grep -q "export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:\\$LD_LIBRARY_PATH" ~/.bashrc || echo \'export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH\' >> ~/.bashrc) && echo "✓ Environment variables added to ~/.bashrc. Restart your terminal or run: source ~/.bashrc"',
    description: 'Permanently adds tt-metal environment variables to ~/.bashrc (idempotent - safe to run multiple times)',
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
    template: 'hf auth login --token "$HF_TOKEN"',
    description: 'Authenticates with Hugging Face using your token',
  },

  // Model Download
  DOWNLOAD_MODEL: {
    id: 'download-model',
    name: 'Download Llama Model',
    template: (() => {
      const model = getDefaultModel();
      return `mkdir -p ~/models && hf download ${model.huggingfaceId} --local-dir ~/models/${model.localDirName}`;
    })(),
    description: (() => {
      const model = getDefaultModel();
      return `Creates ~/models directory and downloads ${model.displayName} model (full model with all formats, ~16GB)`;
    })(),
  },

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
  RUN_INFERENCE: {
    id: 'run-inference',
    name: 'Run Llama Inference',
    template:
      'cd "{{ttMetalPath}}" && export HF_MODEL="meta-llama/Llama-3.1-8B-Instruct" && export LLAMA_DIR="{{modelPath}}" && export PYTHONPATH=$(pwd) && pytest models/tt_transformers/demo/simple_text_demo.py -k performance-batch-1 --max_seq_len 1024 --max_generated_tokens 128',
    description: 'Runs Llama inference demo with HF_MODEL and LLAMA_DIR set to the downloaded model',
    variables: ['ttMetalPath', 'modelPath'],
  },

  // Interactive Chat (Lesson 4)
  INSTALL_INFERENCE_DEPS: {
    id: 'install-inference-deps',
    name: 'Install Inference Dependencies',
    template: 'pip install pi && pip install git+https://github.com/tenstorrent/llama-models.git@tt_metal_tag',
    description: 'Installs pi package and llama-models from Tenstorrent GitHub for inference',
  },

  CREATE_CHAT_SCRIPT: {
    id: 'create-chat-script',
    name: 'Create Interactive Chat Script',
    template: 'mkdir -p ~/tt-scratchpad && cp "{{templatePath}}" ~/tt-scratchpad/tt-chat.py && chmod +x ~/tt-scratchpad/tt-chat.py',
    description: 'Copies the chat script template to ~/tt-scratchpad and makes it executable',
    variables: ['templatePath'],
  },

  START_CHAT_SESSION: {
    id: 'start-chat-session',
    name: 'Start Interactive Chat',
    template:
      'cd "{{ttMetalPath}}" && export LLAMA_DIR="{{modelPath}}" && export PYTHONPATH=$(pwd) && python3 ~/tt-scratchpad/tt-chat.py',
    description: 'Starts the interactive chat REPL with the Llama model on tt-metal',
    variables: ['ttMetalPath', 'modelPath'],
  },

  // HTTP API Server (Lesson 5)
  CREATE_API_SERVER: {
    id: 'create-api-server',
    name: 'Create API Server Script',
    template: 'mkdir -p ~/tt-scratchpad && cp "{{templatePath}}" ~/tt-scratchpad/tt-api-server.py && chmod +x ~/tt-scratchpad/tt-api-server.py',
    description: 'Copies the API server script template to ~/tt-scratchpad and makes it executable',
    variables: ['templatePath'],
  },

  INSTALL_FLASK: {
    id: 'install-flask',
    name: 'Install Flask',
    template: 'pip install flask',
    description: 'Installs Flask web framework for the API server',
  },

  START_API_SERVER: {
    id: 'start-api-server',
    name: 'Start API Server',
    template:
      'cd "{{ttMetalPath}}" && export LLAMA_DIR="{{modelPath}}" && export PYTHONPATH=$(pwd) && python3 ~/tt-scratchpad/tt-api-server.py --port 8080',
    description: 'Starts the Flask API server with the Llama model on tt-metal',
    variables: ['ttMetalPath', 'modelPath'],
  },

  TEST_API_BASIC: {
    id: 'test-api-basic',
    name: 'Test API with Basic Query',
    template:
      'curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d \'{"prompt": "What is machine learning?"}\'',
    description: 'Tests the API server with a basic curl request',
  },

  TEST_API_MULTIPLE: {
    id: 'test-api-multiple',
    name: 'Test API with Multiple Queries',
    template:
      'echo "Testing Tenstorrent query..." && curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d \'{"prompt": "Tell me about Tenstorrent hardware"}\' && echo "\n\nTesting haiku..." && curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d \'{"prompt": "Write a haiku about AI"}\'',
    description: 'Tests the API server with multiple sequential curl requests',
  },

  // ========================================
  // Lesson 6: Production Inference with tt-inference-server
  // ========================================

  VERIFY_INFERENCE_SERVER_PREREQS: {
    id: 'verify-inference-server-prereqs',
    name: 'Verify tt-inference-server Prerequisites',
    template: 'echo "=== Checking Prerequisites ===" && which docker && ls ~/.local/lib/tt-inference-server/run.py && tt-smi && echo "=== ✓ All prerequisites OK ==="',
    description: 'Verifies Docker is installed, tt-inference-server run.py exists, and hardware is detected',
  },

  START_TT_INFERENCE_SERVER: {
    id: 'start-tt-inference-server',
    name: 'Start tt-inference-server (Basic)',
    template: 'cd ~/.local/lib/tt-inference-server && python3 run.py --model Llama-3.1-8B-Instruct --device n150 --workflow server --docker-server',
    description: 'Starts vLLM server via tt-inference-server for Llama 3.1 8B on N150',
  },

  START_TT_INFERENCE_SERVER_N150: {
    id: 'start-tt-inference-server-n150',
    name: 'Start tt-inference-server (N150 Config)',
    template: 'cd ~/.local/lib/tt-inference-server && python3 run.py --model Llama-3.1-8B-Instruct --device n150 --workflow server --docker-server',
    description: 'Starts vLLM server via tt-inference-server optimized for N150 hardware',
  },

  START_TT_INFERENCE_SERVER_N300: {
    id: 'start-tt-inference-server-n300',
    name: 'Start tt-inference-server (N300 Config)',
    template: 'cd ~/.local/lib/tt-inference-server && python3 run.py --model Llama-3.1-8B-Instruct --device n300 --workflow server --docker-server',
    description: 'Starts vLLM server via tt-inference-server optimized for N300 dual-chip hardware',
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

  // Coding Assistant with Prompt Engineering (Lesson 9)
  VERIFY_CODING_MODEL: {
    id: 'verify-coding-model',
    name: 'Verify Llama 3.1 8B',
    template: 'ls -lh ~/models/Llama-3.1-8B-Instruct/original/',
    description: 'Verifies Llama 3.1 8B model is downloaded (should be from Lesson 3)',
  },

  CREATE_CODING_ASSISTANT_SCRIPT: {
    id: 'create-coding-assistant-script',
    name: 'Create Coding Assistant Script',
    template: 'mkdir -p ~/tt-scratchpad',
    description: 'Creates the coding assistant script with prompt engineering in ~/tt-scratchpad',
  },

  START_CODING_ASSISTANT: {
    id: 'start-coding-assistant',
    name: 'Start Coding Assistant',
    template: 'cd ~/tt-metal && export LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original && export PYTHONPATH=$(pwd) && python3 ~/tt-scratchpad/tt-coding-assistant.py',
    description: 'Starts interactive CLI coding assistant with Llama 3.1 8B using Direct API and prompt engineering',
  },

  // Image Classification with TT-Forge (Lesson 11)
  // forge + torch-xla are pre-installed in /opt/venv-forge — activate and go.
  BUILD_FORGE_FROM_SOURCE: {
    id: 'build-forge-from-source',
    name: 'Activate Forge Environment',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import forge, jax; print(\'forge:\', forge.__version__); print(\'jax:\', jax.__version__); print(\'devices:\', jax.devices())"',
    description: 'Activates the pre-installed venv-forge environment and verifies the forge stack',
  },

  INSTALL_FORGE: {
    id: 'install-forge',
    name: 'Activate Forge Environment',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import forge, jax; print(\'forge:\', forge.__version__); print(\'jax:\', jax.__version__); print(\'devices:\', jax.devices())"',
    description: 'Activates the pre-installed venv-forge environment (forge is pre-installed, no pip needed)',
  },

  TEST_FORGE_INSTALL: {
    id: 'test-forge-install',
    name: 'Verify Forge Stack',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import forge, jax, torch_xla; print(\'forge    :\', forge.__version__); print(\'jax      :\', jax.__version__); print(\'torch_xla:\', torch_xla.__version__); print(\'tt devices:\', jax.devices())"',
    description: 'Imports forge, jax, and torch_xla and prints their versions + visible TT devices',
  },

  TEST_FORGE_INSTALL_WHEEL: {
    id: 'test-forge-install-wheel',
    name: 'Verify Forge Stack',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import forge, jax, torch_xla; print(\'forge    :\', forge.__version__); print(\'jax      :\', jax.__version__); print(\'torch_xla:\', torch_xla.__version__); print(\'tt devices:\', jax.devices())"',
    description: 'Imports forge, jax, and torch_xla and prints their versions + visible TT devices',
  },

  CREATE_FORGE_CLASSIFIER: {
    id: 'create-forge-classifier',
    name: 'Create Image Classifier Script',
    template: 'mkdir -p ~/tt-scratchpad && cp \"{{templatePath}}\" ~/tt-scratchpad/tt-forge-classifier.py && chmod +x ~/tt-scratchpad/tt-forge-classifier.py',
    description: 'Copies tt-forge-classifier.py template to ~/tt-scratchpad',
    variables: ['templatePath'],
  },

  RUN_FORGE_CLASSIFIER: {
    id: 'run-forge-classifier',
    name: 'Run Image Classifier',
    template: 'source /etc/profile.d/tt-env-forge.sh && cd ~/tt-scratchpad && python3 tt-forge-classifier.py',
    description: 'Activates venv-forge and runs MobileNetV2 image classification on TT hardware',
  },

  RUN_FORGE_CLASSIFIER_WHEEL: {
    id: 'run-forge-classifier-wheel',
    name: 'Run Image Classifier (legacy)',
    template: 'source /etc/profile.d/tt-env-forge.sh && cd ~/tt-scratchpad && python3 tt-forge-classifier.py',
    description: 'Activates venv-forge and runs MobileNetV2 image classifier (alias)',
  },

  RUN_FORGE_CUSTOM_IMAGE: {
    id: 'run-forge-custom-image',
    name: 'Classify Custom Image',
    template: 'source /etc/profile.d/tt-env-forge.sh && cd ~/tt-scratchpad && python3 tt-forge-classifier.py --image {{imagePath}}',
    description: 'Classifies a user-provided image with TT-Forge compiled model',
    variables: ['imagePath'],
  },

  // ──────────────────────────────────────────────────────────────────────────
  // venv-forge activation & verification (shared by forge + XLA lessons)
  // ──────────────────────────────────────────────────────────────────────────

  ACTIVATE_FORGE_ENV: {
    id: 'activate-forge-env',
    name: 'Activate Forge Environment',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import jax; print(\'TT devices:\', jax.devices())"',
    description: 'Activates the pre-installed venv-forge environment and prints visible TT devices',
  },

  VERIFY_FORGE_STACK: {
    id: 'verify-forge-stack',
    name: 'Verify Forge Stack',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import forge, jax, torch_xla; print(\'forge    :\', forge.__version__); print(\'jax      :\', jax.__version__); print(\'torch_xla:\', torch_xla.__version__); print(\'tt devices:\', jax.devices())"',
    description: 'Imports forge, jax, and torch_xla and prints their versions + visible TT devices',
  },

  // TT-XLA JAX Integration
  RUN_JAX_QUICKSTART: {
    id: 'run-jax-quickstart',
    name: 'Run JAX Quickstart',
    template: "source /etc/profile.d/tt-env-forge.sh && python3 - <<'PYEOF'\nimport jax\nimport jax.numpy as jnp\na = jnp.ones((1024, 1024))\nb = jnp.ones((1024, 1024))\nc = a @ b\nprint('shape  :', c.shape)\nprint('device :', c.devices())\nprint('c[0,0] :', c[0, 0])  # expect 1024.0\nPYEOF",
    description: 'Runs a 1024x1024 JAX matmul on TT hardware via venv-forge; prints shape, device, value',
  },

  RUN_JAX_PMAP_DEMO: {
    id: 'run-jax-pmap-demo',
    name: 'Run JAX pmap Demo (multi-device)',
    template: "source /etc/profile.d/tt-env-forge.sh && python3 - <<'PYEOF'\nimport jax\nimport jax.numpy as jnp\ndevices = jax.devices()\nn = len(devices)\nprint(f'Running across {n} TT device(s)')\n@jax.pmap\ndef matmul_per_device(A):\n    return A @ A.T\nA = jnp.ones((n, 512, 512))\nresult = matmul_per_device(A)\nprint('result shape :', result.shape)\nprint('sharding     :', result.sharding)\nPYEOF",
    description: 'Maps a matmul across all TT devices using jax.pmap (QB2 uses all 4 chips)',
  },

  RUN_PYTORCH_XLA_DEMO: {
    id: 'run-pytorch-xla-demo',
    name: 'Run PyTorch/XLA Demo',
    template: "source /etc/profile.d/tt-env-forge.sh && python3 - <<'PYEOF'\nimport torch\nimport torch_xla.core.xla_model as xm\ndevice = xm.xla_device()\nprint('TT device:', device)\nx = torch.randn(256, 256).to(device)\ny = torch.randn(256, 256).to(device)\nz = x @ y\nxm.mark_step()\nprint('z.shape :', z.shape)\nprint('z.device:', z.device)\nPYEOF",
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
    template: 'source /etc/profile.d/tt-env-forge.sh && cd ~/tt-forge/demos/tt-xla/nlp/jax && pip install -q -r requirements.txt && python3 gpt_demo.py',
    description: 'Runs official GPT-2 next-token demo via JAX on TT hardware',
  },

  // These commands keep the registered handlers for installTtXla and testTtXlaInstall
  // wired up (they display a helpful redirect rather than doing nothing).
  INSTALL_TT_XLA: {
    id: 'install-tt-xla',
    name: 'Activate Forge Environment',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import jax; print(\'TT devices:\', jax.devices())"',
    description: 'venv-forge is pre-installed — activates the environment instead',
  },

  CREATE_TT_XLA_TEST: {
    id: 'create-tt-xla-test',
    name: 'Run JAX Quickstart',
    template: "source /etc/profile.d/tt-env-forge.sh && python3 - <<'PYEOF'\nimport jax, jax.numpy as jnp\nc = jnp.ones((1024, 1024)) @ jnp.ones((1024, 1024))\nprint('shape:', c.shape, 'device:', c.devices())\nPYEOF",
    description: 'Runs a quick JAX matmul to verify TT-XLA is working',
  },

  TEST_TT_XLA_INSTALL: {
    id: 'test-tt-xla-install',
    name: 'Verify Forge Stack',
    template: 'source /etc/profile.d/tt-env-forge.sh && python3 -c "import jax; print(jax.devices())"',
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

  INSTALL_ANIMATEDIFF: {
    id: 'install-animatediff',
    name: 'Install AnimateDiff Package',
    template: 'cd ~/tt-scratchpad/tt-animatediff && pip install -e . && python3 -c "import animatediff_ttnn; print(\'✓ AnimateDiff package installed successfully\')"',
    description: 'Install the animatediff-ttnn standalone package',
  },

  RUN_ANIMATEDIFF_2FRAME: {
    id: 'run-animatediff-2frame',
    name: 'Run 2-Frame Demo',
    template: 'cd ~/tt-scratchpad/tt-animatediff && python3 examples/generate_2frame_video.py 2>&1 | grep -v "DEBUG\\|Config{"',
    description: 'Test temporal attention with minimal 2-frame sequence',
  },

  RUN_ANIMATEDIFF_16FRAME: {
    id: 'run-animatediff-16frame',
    name: 'Run 16-Frame Demo',
    template: 'cd ~/tt-scratchpad/tt-animatediff && python3 examples/generate_16frame_video.py 2>&1 | grep -v "DEBUG\\|Config{"',
    description: 'Generate full 16-frame animated sequence with temporal coherence',
  },

  VIEW_ANIMATEDIFF_OUTPUT: {
    id: 'view-animatediff-output',
    name: 'View Generated Animation',
    template: 'ls -lh ~/tt-scratchpad/tt-animatediff/output/test_16frame.gif && echo "\n✓ Animation generated: ~/tt-scratchpad/tt-animatediff/output/test_16frame.gif"',
    description: 'View the generated 16-frame animation file',
  },

  SETUP_ANIMATEDIFF_PROJECT: {
    id: 'setup-animatediff-project',
    name: 'Setup AnimateDiff Project',
    template: 'mkdir -p ~/tt-scratchpad/tt-animatediff && cp -r "{{projectPath}}"/* ~/tt-scratchpad/tt-animatediff/ && cd ~/tt-scratchpad/tt-animatediff && pip install -e . && python3 -c "import animatediff_ttnn; print(\'✓ AnimateDiff project setup complete at ~/tt-scratchpad/tt-animatediff/\')"',
    description: 'Copies AnimateDiff project from extension to ~/tt-scratchpad/tt-animatediff and installs it',
    variables: ['projectPath'],
  },

  GENERATE_ANIMATEDIFF_VIDEO_SD35: {
    id: 'generate-animatediff-video-sd35',
    name: 'Generate Animated Video with SD 3.5',
    template: 'cd ~/tt-scratchpad/tt-animatediff && python3 examples/generate_with_sd35.py 2>&1 | grep -v "DEBUG\\|Config{"',
    description: 'Generate animated video using SD 3.5 + AnimateDiff temporal attention (GNU cinemagraph)',
  },

  // ========================================
  // Custom Training Lessons
  // ========================================

  INSTALL_TT_TRAIN: {
    id: 'install-tt-train',
    name: 'Install tt-train',
    template: 'cd $TT_METAL_HOME/tt-train && pip install -e . && echo "✓ tt-train installed successfully"',
    description: 'Install tt-train Python package from tt-metal repository',
  },

  // CT-8: Training from Scratch Commands
  PREPARE_SHAKESPEARE: {
    id: 'prepare-shakespeare',
    name: 'Prepare Shakespeare Dataset',
    template: 'cd ~/tt-scratchpad/training/data && python prepare_shakespeare.py --output shakespeare.txt --split',
    description: 'Download and prepare tiny-shakespeare dataset for training from scratch',
  },

  CREATE_NANO_TRICKSTER: {
    id: 'create-nano-trickster',
    name: 'Create Nano-Trickster Architecture',
    template: 'cd ~/tt-scratchpad/training && python nano_trickster.py',
    description: 'Test the nano-trickster architecture (11M parameters)',
  },

  TRAIN_FROM_SCRATCH: {
    id: 'train-from-scratch',
    name: 'Train from Scratch',
    template: 'cd ~/tt-scratchpad/training && python train_from_scratch.py --config configs/nano_trickster.yaml',
    description: 'Train nano-trickster from random initialization on tiny-shakespeare',
  },

  TEST_NANO_TRICKSTER: {
    id: 'test-nano-trickster',
    name: 'Test Nano-Trickster',
    template: 'cd ~/tt-scratchpad/training && python -c "import torch; from nano_trickster import NanoTrickster; model = NanoTrickster(); model.load_state_dict(torch.load(\'output/nano_trickster/final_model.pt\')); model.eval(); tokenizer = torch.load(\'data/tokenizer.pt\'); stoi = tokenizer[\'stoi\']; itos = tokenizer[\'itos\']; prompt = \'ROMEO:\'; input_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]]); generated = model.generate(input_ids, max_new_tokens=200, temperature=0.8); text = \'\'.join([itos.get(int(t), \'?\') for t in generated[0]]); print(text)"',
    description: 'Generate text with the trained nano-trickster model',
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
