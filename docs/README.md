# VulkanIlm 🚀🔥 – GPU-Accelerated Local LLMs for Everyone

![GitHub stars](https://img.shields.io/github/stars/ggerganov/llama.cpp?style=social)
![GitHub forks](https://img.shields.io/github/forks/ggerganov/llama.cpp?style=social)
![GitHub license](https://img.shields.io/github/license/Talnz007/VulkanIlm)
![Python version](https://img.shields.io/badge/python-3.9+-blue.svg)
![GitHub issues](https://img.shields.io/github/issues/Talnz007/VulkanIlm)

**VulkanIlm** (from _"Vulkan"_ 🔥 + _"Ilm"_ 📚, meaning "knowledge" in Urdu/Arabic) is a Python library that brings GPU-accelerated local LLM inference to **AMD and Intel GPU users**—no CUDA required.

Built specifically for developers with legacy GPUs, VulkanIlm enables blazing-fast local inference using [`llama.cpp`](https://github.com/ggerganov/llama.cpp)'s Vulkan backend.

---

## 🎯 Why VulkanIlm?

> **Not everyone has an NVIDIA GPU. Everyone deserves fast local AI.**

**The Problem:** Most GPU acceleration libraries focus on CUDA, leaving AMD/Intel users with slow CPU-only inference.

**The Solution:** VulkanIlm democratizes GPU-accelerated AI for **all GPU brands** using Vulkan.

### Key Features

- 🚀 **4-6x faster** than CPU inference on legacy GPUs
- 🎮 **Universal GPU support**: AMD, Intel, and NVIDIA
- 🐍 **Python-first** with simple CLI tools
- ⚡ **Auto-detection** and optimization for your GPU
- 📦 **Zero-config installation** with auto-building
- 🔄 **Real-time streaming** token generation

---

## 📊 Performance Results

| Hardware | CPU Performance | Vulkan Performance | **Speedup** |
|----------|-----------------|-------------------|-------------|
| AMD RX 580 8GB | 188.47s | 44.74s | **4.21x** |
| Intel Arc A770 | ~120s | ~25s | **4.8x** |
| AMD RX 6600 | ~90s | ~18s | **5.0x** |

*Benchmarked with Gemma-3n-E4B-it model (6.9B parameters)*

---

## 📦 Installation

### Quick Start

```bash
git clone https://github.com/Talnz007/VulkanIlm.git
cd VulkanIlm
pip install -e .
```

### Prerequisites

- **Python 3.9+**
- **Vulkan-capable GPU** (AMD RX 400+, Intel Arc/Xe, NVIDIA GTX 900+)
- **Vulkan drivers** installed

### Install Vulkan Drivers (if needed)

```bash
# Ubuntu/Debian
sudo apt install vulkan-tools libvulkan-dev

# Fedora/RHEL
sudo dnf install vulkan-tools vulkan-devel

# Test installation
vulkaninfo
```

---

## 🚀 Usage

### CLI Interface

```bash
# Auto-install llama.cpp with Vulkan support
vulkanilm install

# Check your GPU setup
vulkanilm vulkan-info

# Download and use models
vulkanilm search "llama"
vulkanilm download microsoft/DialoGPT-medium

# Generate text
vulkanilm ask model.gguf --prompt "Explain quantum computing"

# Stream in real-time
vulkanilm stream model.gguf "Tell me a story about AI"

# Benchmark CPU vs GPU performance
vulkanilm benchmark "Test prompt"
```

### Python API

```python
from vulkan_ilm import Llama

# Load model with automatic GPU optimization
llm = Llama("path/to/model.gguf", gpu_layers=16)

# Generate text
response = llm.ask("Explain the term 'ilm' in AI context.")
print(response)

# Stream tokens in real-time
for token in llm.stream_ask_real("Tell me about Vulkan API"):
    print(token, end='', flush=True)
```

---

## 🎮 Supported Hardware

### Tested GPUs

| Brand | Models | Status | Performance |
|-------|--------|--------|-------------|
| **AMD** | RX 580, RX 590, RX 6600, RX 6700 | ✅ Excellent | 4-5x speedup |
| **Intel** | Arc A770, Arc A750, Xe Graphics | ✅ Great | 4-5x speedup |
| **NVIDIA** | GTX 1060+, RTX series | ✅ Excellent | 4-6x speedup |

---

## 🛠️ Troubleshooting

### Common Issues

**❌ `vulkanilm: command not found`**

```bash
# Ensure you installed the package correctly
pip install -e .
poetry install  # if using Poetry

# Check if CLI is available
which vulkanilm
```

**❌ `No Vulkan support detected`**

```bash
# Install Vulkan tools and test
sudo apt install vulkan-tools libvulkan-dev
vulkaninfo

# Update GPU drivers
```

**❌ `Model hangs or uses too much VRAM`**

```bash
# Use fewer GPU layers
vulkanilm ask model.gguf --gpu-layers 8 -p "test"

# Or force CPU mode for testing
vulkanilm ask model.gguf --cpu -p "test"
```

---

## 🏗️ Architecture

```
VulkanIlm/
├── vulkan_ilm/
│   ├── cli.py              # Command-line interface
│   ├── llama.py            # Main Python API
│   ├── vulkan/
│   │   └── detector.py     # GPU detection & optimization
│   ├── benchmark.py        # Performance testing
│   ├── installer.py        # Auto-build llama.cpp
│   └── streaming.py        # Real-time token streaming
├── pyproject.toml          # Poetry configuration
└── README.md
```

---

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- **GPU Testing**: Test on different AMD/Intel/NVIDIA cards
- **Model Support**: Add support for new model formats
- **Performance**: Optimize memory usage and speed
- **Documentation**: Improve guides and examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 🧾 The Story Behind "VulkanIlm"

In South Asian and Islamic culture, **"Ilm" (علم)** represents *knowledge*, wisdom, and enlightenment.

Combined with **Vulkan**—a high-performance GPU API—this project embodies **"knowledge on fire"** 🔥: making advanced AI accessible to everyone, regardless of their GPU brand or budget.

**Our mission:** Democratize local AI inference for the global developer community.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📞 Support & Links

- **GitHub**: [https://github.com/Talnz007/VulkanIlm](https://github.com/Talnz007/VulkanIlm)
- **Issues**: [Report bugs or request features](https://github.com/Talnz007/VulkanIlm/issues)
- **Discussions**: [Community Q&A](https://github.com/Talnz007/VulkanIlm/discussions)

---

> **🔥 Built with passion by [@Talnz007](https://github.com/Talnz007) — Bringing fast, local AI to legacy GPUs everywhere**
