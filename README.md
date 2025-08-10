VulkanIlm 🚀🔥 – GPU-Accelerated Local LLMs for Everyone

VulkanIlm (from "Vulkan" 🔥 + "Ilm" 📚, meaning "knowledge" in Urdu/Arabic) is a Python library that brings GPU-accelerated local LLM inference to AMD and Intel GPU users—no CUDA required.

Built specifically for developers with legacy GPUs, VulkanIlm enables blazing-fast local inference using llama.cpp's](https://github.com/ggerganov/llama.cpp)'s) Vulkan backend.
🎯 Why VulkanIlm?

    Not everyone has an NVIDIA GPU. Everyone deserves fast local AI.

The Problem: Most GPU acceleration libraries focus on CUDA, leaving AMD/Intel users with slow CPU-only inference.

The Solution: VulkanIlm democratizes GPU-accelerated AI for all GPU brands using Vulkan.
Key Features

    🚀 4-6x faster than CPU inference on legacy GPUs

    🎮 Universal GPU support: AMD, Intel, and NVIDIA

    🐍 Python-first with simple CLI tools

    ⚡ Auto-detection and optimization for your GPU

    📦 Zero-config installation with auto-building

    🔄 Real-time streaming token generation

📊 Performance Results

Hardware
	

CPU Performance
	

Vulkan Performance
	

Speedup

AMD RX 580 8GB
	

188.47s
	

44.74s
	

4.21x

Intel Arc A770
	

~120s
	

~25s
	

4.8x

AMD RX 6600
	

~90s
	

~18s
	

5.0x

Benchmarked with Gemma-3n-E4B-it model (6.9B parameters)
📦 Installation
Quick Start

git clone https://github.com/Talnz007/VulkanIlm.git
cd VulkanIlm
pip install -e .

Prerequisites

    Python 3.9+

    Vulkan-capable GPU (AMD RX 400+, Intel Arc/Xe, NVIDIA GTX 900+)

    Vulkan drivers installed

Install Vulkan Drivers (if needed)

# Ubuntu/Debian
sudo apt install vulkan-tools libvulkan-dev

# Fedora/RHEL
sudo dnf install vulkan-tools vulkan-devel

# Test installation
vulkaninfo

🚀 Usage
CLI Interface

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

Python API

from vulkan_ilm import Llama

# Load model with automatic GPU optimization
llm = Llama("path/to/model.gguf", gpu_layers=16)

# Generate text
response = llm.ask("Explain the term 'ilm' in AI context.")
print(response)

# Stream tokens in real-time
for token in llm.stream_ask_real("Tell me about Vulkan API"):
    print(token, end='', flush=True)

🎮 Supported Hardware
Tested GPUs

Brand
	

Models
	

Status
	

Performance

AMD
	

RX 580, RX 590, RX 6600, RX 6700
	

✅ Excellent
	

4-5x speedup

Intel
	

Arc A770, Arc A750, Xe Graphics
	

✅ Great
	

4-5x speedup

NVIDIA
	

GTX 1060+, RTX series
	

✅ Excellent
	

4-6x speedup
🛠️ Troubleshooting Common Installation Issues (Linux)

Here are some common errors you might encounter during installation and how to resolve them.
❌ vulkanilm: command not found

This error means your shell cannot find the vulkanilm executable. It's often caused by the pip install -e . command not properly setting up the symbolic link or your virtual environment not being activated.

Solution:

First, ensure you are in an activated virtual environment.

# To create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Then, re-run the installation
pip install -e .

If you are using poetry as indicated in pyproject.toml, you can run the command directly through poetry to avoid path issues:

poetry run vulkanilm install

❌ Could NOT find Vulkan (missing: glslc)

This error occurs when the cmake build process for llama.cpp cannot find the glslc shader compiler, which is a key part of the Vulkan SDK.

Solution:

You need to install the package that provides glslc.

    On Fedora/RHEL:

    sudo dnf install glslc

    On Ubuntu/Debian:

    # `glslc` is typically part of the vulkan-tools or vulkan-sdk package
    sudo apt install vulkan-tools

After installation, verify that the command is available:

glslc --version

❌ Could NOT find CURL

This error happens when the cmake build process can't locate the libcurl development libraries, which are required for llama.cpp to download models.

Solution:

Install the development package for libcurl.

    On Fedora/RHEL:

    sudo dnf install libcurl-devel

    On Ubuntu/Debian:

    sudo apt install libcurl4-openssl-dev

🏗️ Architecture

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

🤝 Contributing

We welcome contributions! Areas where you can help:

    GPU Testing: Test on different AMD/Intel/NVIDIA cards

    Model Support: Add support for new model formats

    Performance: Optimize memory usage and speed

    Documentation: Improve guides and examples

See CONTRIBUTING.md for details.
🧾 The Story Behind "VulkanIlm"

In South Asian and Islamic culture, "Ilm" (علم) represents knowledge, wisdom, and enlightenment.

Combined with Vulkan—a high-performance GPU API—this project embodies "knowledge on fire" 🔥: making advanced AI accessible to everyone, regardless of their GPU brand or budget.

Our mission: Democratize local AI inference for the global developer community.
📄 License

MIT License - see LICENSE for details.
📞 Support & Links

    GitHub: https://github.com/Talnz007/VulkanIlm

    Issues: Report bugs or request features

    Discussions: Community Q&A

    🔥 Built with passion by @Talnz007 — Bringing fast, local AI to legacy GPUs everywhere
