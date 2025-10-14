# VulkanIlm 🚀🔥
[![PyPI version](https://badge.fury.io/py/vulkan-ilm.svg)](https://pypi.org/project/vulkan-ilm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://talnz007.github.io/VulkanIlm/#/)
[![GitHub issues](https://img.shields.io/github/issues/Talnz007/VulkanIlm)](https://github.com/Talnz007/VulkanIlm/issues)

**GPU-Accelerated Local LLMs for Everyone (Vulkan + Ilm — "knowledge")**

VulkanIlm is a Python-first wrapper and CLI around `llama.cpp`'s **Vulkan** backend that brings fast local LLM inference to AMD, Intel, and NVIDIA GPUs — **no CUDA required**. Built for developers with legacy or non-NVIDIA hardware.

---

## TL;DR
- **What:** Python library + CLI to run LLMs locally using Vulkan GPU acceleration.  
- **Why:** Most acceleration tooling targets CUDA/NVIDIA — VulkanIlm opens up AMD & Intel users.  
- **Quick result:** Small models can run **orders of magnitude faster** on iGPUs; mid/large legacy GPUs get **~4–6×** speedups vs CPU.

---

## Key features
- 🚀 Significant speedups vs CPU on legacy GPUs and iGPUs  
- 🎮 Broad GPU support: AMD, Intel, NVIDIA (via Vulkan)  
- 🐍 Python-first API + easy CLI tools  
- ⚡ Auto detection + GPU-specific optimizations  
- 📦 Auto build/install of `llama.cpp` Vulkan backend  
- 🔄 Real-time streaming token generation  
- ✅ Reproducible benchmark scripts in `benchmarks/`

---

## Benchmarks (summary)
> Benchmarks measured with Gemma-3n-E4B-it (6.9B) unless noted. Results depend on model quantization, GPU drivers, OS, and system load.

| Hardware (OS) | Model | CPU time | Vulkan (GPU) time | Speedup |
|---|---:|---:|---:|---:|
| **Dell E7250 (i7-5600U, integrated GPU)** — Fedora 42 Workstation | TinyLLaMA-1.1B-Chat (Q4_K_M) | **121 s** | **3 s** | **33×** |
| AMD RX 580 8GB — Ubuntu 22.04.5 LTS (Jammy) | Gemma-3n-E4B-it (6.9B) | 188.47 s | 44.74 s | 4.21× |
| Intel Arc A770 | Gemma-3n-E4B-it (6.9B) | ~120 s | ~25 s | ~4.8× |
| AMD RX 6600 | Gemma-3n-E4B-it (6.9B) | ~90 s | ~18 s | ~5.0× |

**iGPU notes**
- The Dell E7250 iGPU result shows older integrated GPUs can be *very* effective for smaller LLMs when using Vulkan.  
- Smaller models and appropriate quantizations are more iGPU-friendly. Driver/version differences significantly affect results.

**Other tested (functional) models**
- `DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit` — runs (not benchmarked).  
- `LLaMA 3.1 8B` — runs (not benchmarked).

---

## ROCm / AMD notes
- **ROCm is not officially supported** for `gfx803` (RX 580).  
- Some community members try ROCm 5/6 workarounds on RX 580, but they are unstable/unsupported.  
- VulkanIlm offers a Vulkan-based path that avoids ROCm on legacy AMD cards.

---

## Install

**Quick start**
```bash
git clone https://github.com/Talnz007/VulkanIlm.git
cd VulkanIlm
pip install -e .
```

**Prerequisites**

* Python 3.9+
* Vulkan-capable GPU (AMD RX 400+, Intel Arc/Xe, NVIDIA GTX 900+)
* Vulkan drivers installed and working

**Install Vulkan tools (if needed)**

Ubuntu / Debian:

```bash
sudo apt update
sudo apt install vulkan-tools libvulkan-dev
```

Fedora / RHEL:

```bash
sudo dnf install vulkan-tools vulkan-devel
```

Verify:

```bash
vulkaninfo
```

---

## Usage

### CLI examples

```bash
# Auto-install llama.cpp with Vulkan support
vulkanilm install

# Check your GPU setup
vulkanilm vulkan-info

# Search and download models (if supported)
vulkanilm search "llama"
vulkanilm download microsoft/DialoGPT-medium

# Generate text
vulkanilm ask path/to/model.gguf --prompt "Explain quantum computing"

# Stream tokens in real-time
vulkanilm stream path/to/model.gguf "Tell me a story about AI"

# Run a benchmark
vulkanilm benchmark path/to/model.gguf --prompt "Benchmark prompt" --repeat 3
```

### Python API (example)

```python
from vulkan_ilm import Llama

# Load model (auto GPU optimization)
llm = Llama("path/to/model.gguf", gpu_layers=16)

# Synchronous generation
response = llm.ask("Explain the term 'ilm' in AI context.")
print(response)

# Streaming generation
for token in llm.stream_ask_real("Tell me about Vulkan API"):
    print(token, end='', flush=True)
```

---

## Reproduce benchmarks (quick checklist)

1. Use the exact model file & quantization referenced in `/benchmarks` (GGUF + quantization).
2. Use the benchmark script in `benchmarks/run_benchmark.sh`.
3. Record: driver version, OS version, CPU frequency governor, and system load.
4. Run benchmarks multiple times (cold and warm cache) and average results.

---

## Troubleshooting (Linux)

### `vulkanilm: command not found`

* Activate venv and reinstall:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

* Or run via Poetry:

```bash
poetry run vulkanilm install
```

### `Could NOT find Vulkan (missing: glslc)`

* Install `glslc` (Vulkan SDK / vulkan-tools):

```bash
# Fedora
sudo dnf install glslc

# Ubuntu/Debian
sudo apt install vulkan-tools
```

Verify: `glslc --version`

### `Could NOT find CURL`

* Install libcurl dev:

```bash
# Fedora
sudo dnf install libcurl-devel

# Ubuntu/Debian
sudo apt install libcurl4-openssl-dev
```

---

## Project structure

```
VulkanIlm/
├── vulkan_ilm/
│   ├── cli.py
│   ├── llama.py
│   ├── vulkan/
│   │   └── detector.py
│   ├── benchmark.py
│   ├── installer.py
│   └── streaming.py
├── benchmarks/             # benchmark scripts & data
├── pyproject.toml
└── README.md
```

---

## Contributing

We welcome contributions! Useful areas:

* GPU testing across drivers & OSes
* Additional model formats & quant recipes
* Memory & perf optimizations
* Docs, reproducible benchmarks, and examples

See `CONTRIBUTING.md` for details. Look for `good-first-issue` tags.

---

## The story behind the name

*Ilm* (علم) = knowledge / wisdom. Combined with Vulkan — “knowledge on fire”: making fast local AI accessible to everyone, regardless of GPU brand or budget. 🔥

---

## License

MIT — see `LICENSE` for details.

---

## Links & support

* **Repo:** [https://github.com/Talnz007/VulkanIlm](https://github.com/Talnz007/VulkanIlm)
* **Issues:** Report bugs or request features on GitHub
* **Discussions:** Community Q\&A
* 📘 **Full Documentation:** [https://talnz007.github.io/VulkanIlm/#/](https://talnz007.github.io/VulkanIlm/#/)


---

*Built with passion by @Talnz007 — bringing fast, local AI to legacy GPUs everywhere.*
