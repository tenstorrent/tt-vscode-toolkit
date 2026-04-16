---
id: download-model
title: Download Model and Run Inference
description: >-
  Download Qwen3-0.6B (the recommended model — no license gate, works on all
  hardware) from Hugging Face to run AI workloads on your Tenstorrent hardware.
  Optionally download Llama-3.1-8B-Instruct for N300+ hardware.
category: first-inference
tags:
  - hardware
  - inference
  - model
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: validated
validatedOn:
  - n150
  - p300c
estimatedMinutes: 10
---

# Download Model from Hugging Face

Download **Qwen3-0.6B** — the recommended model for Tenstorrent hardware. It's
tiny (0.6B parameters), fast, reasoning-capable, and requires no special license
agreement. Works reliably on every supported device including N150 and P300c.

## Prerequisites

You'll need a **Hugging Face access token** to download models. If you don't
have one:

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up or log in
3. Navigate to Settings → Access Tokens
4. Create a new token with read permissions

---

## Starting Fresh?

If you're jumping directly to this lesson, let's verify your setup:

### Quick Prerequisite Checks

```bash
# Hardware detected?
tt-smi -s

# Python installed?
python3 --version  # Need 3.10+

# hf CLI installed? (included with huggingface-hub)
which hf || pip install huggingface-hub
```

**All checks passed?** Continue below to download the model.

**If hardware check failed:**
- See [Hardware Detection](command:tenstorrent.showLesson?["hardware-detection"]) to set up tt-smi

---

### Already Authenticated?

Check if you're already logged in to Hugging Face:

```bash
hf auth whoami
```

**If this shows your username:** You're already authenticated! Skip to
[Step 3: Download the Model](#step-3-download-qwen3-0-6b).

**If it shows an error:** Continue with Step 1 below to authenticate.

---

### Model Already Downloaded?

Check if Qwen3-0.6B is already present:

```bash
ls ~/models/Qwen3-0.6B/config.json
```

**If the file exists:** Model is already downloaded! You can skip to the next
lesson.

**Verify download contents:**
```bash
ls ~/models/Qwen3-0.6B/config.json
ls ~/models/Qwen3-0.6B/model*.safetensors
```

**If you already have Llama-3.1-8B-Instruct:**
```bash
ls ~/models/Llama-3.1-8B-Instruct/config.json
```

**All files present?** You're good to go!

**Missing files?** Redownload using Step 3 below.

---

## Understanding Model Formats

**Qwen3-0.6B uses HuggingFace format only:**
- Files: `config.json`, `model.safetensors`, `tokenizer.json`
- Used by: vLLM (Lessons 6-7), Direct API (Lessons 4-5)
- Path: `~/models/Qwen3-0.6B/`

The HuggingFace format is the standard for modern models and is compatible with
all Tenstorrent inference tools.

---

## Step 1: Set Your Token

**First, check if your token is already set:**

```bash
echo $HF_TOKEN
```

**If you see your token:** It's already set! Skip to
[Step 2: Authenticate](#step-2-authenticate).

**If it's empty:** Set your token using one of these methods:

### Method 1: Via Extension (Recommended)

When you click the button below, you'll be prompted to enter your token
securely:

[🔑 Enter Your Hugging Face Token](command:tenstorrent.setHuggingFaceToken)

### Method 2: Manually in Terminal

```bash
export HF_TOKEN=your_token_from_huggingface
```

**Note:** This only lasts for your current terminal session. For permanent
setup, add it to `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export HF_TOKEN=your_token_from_huggingface' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 2: Authenticate

Once your token is set, authenticate with Hugging Face:

```bash
hf auth login --token "$HF_TOKEN"
```

[✓ Authenticate with Hugging Face](command:tenstorrent.loginHuggingFace)

---

## Step 3: Download Qwen3-0.6B

> **Tip — custom storage location:** All lessons use `~/models` as the model
> directory. If your models live on a larger drive or shared storage, symlink
> `~/models` there once and every lesson will find them automatically:
> ```bash
> ln -s /path/to/your/storage ~/models
> ```
> Already downloaded a model to the HF cache? Symlink the snapshot:
> ```bash
> mkdir -p ~/models
> ln -sfn "$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/ | tail -1 | xargs -I{} echo ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/{})" ~/models/Qwen3-0.6B
> ```

Download Qwen3-0.6B — no license gate, no terms to accept, works on all
Tenstorrent hardware:

```bash
mkdir -p ~/models && hf download Qwen/Qwen3-0.6B \
  --local-dir ~/models/Qwen3-0.6B
```

[⬇️ Download Qwen3-0.6B Model](command:tenstorrent.downloadQwen3Small)

## What Gets Downloaded

The Qwen3-0.6B model includes:
- **Model weights** — HuggingFace `.safetensors` format
- **Tokenizer files** — `tokenizer.json`, `tokenizer_config.json`
- **Config files** — `config.json`, `generation_config.json`
- **Total size** — approximately 1.2GB

The download typically completes in under a minute on a fast connection.

---

## Optional: Llama-3.1-8B-Instruct (Gated Model)

> **Note:** Meta requires accepting their data license at
> [huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
> before this download will succeed. If you prefer open models or haven't
> accepted Meta's terms, **Qwen3-0.6B is an excellent alternative**.

> **Hardware requirement:** Llama-3.1-8B-Instruct requires **N300 or higher**
> for reliable operation. It consistently exhausts DRAM on N150 and P300c.
> Qwen3-0.6B is the recommended choice for those devices.

If you've accepted Meta's license terms and are running on N300/T3K/P100/Galaxy,
you can download Llama-3.1-8B-Instruct:

```bash
mkdir -p ~/models && hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/models/Llama-3.1-8B-Instruct
```

**What gets downloaded (~16GB):**
- HuggingFace format files — `config.json`, `model.safetensors`, etc. (for vLLM)
- Meta original format files — `params.json`, `consolidated.00.pth`,
  `tokenizer.model` (in `original/` subdirectory, for Direct API)

---

## Next Steps

You've successfully downloaded your model and are ready to run inference.

**Next: Verify Your Setup →** [verify-installation](command:tenstorrent.showLesson?["verify-installation"])

## Learn More

- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Llama 3.1 model on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [TT-Metal GitHub Repository](https://github.com/tenstorrent/tt-metal)
- [TT-NN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/)
