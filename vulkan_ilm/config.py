# vulkan_ilm/config.py
"""
Configuration file for VulkanIlm.
"""
import os
from pathlib import Path

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the llama.cpp binary - use the exact path you found
DEFAULT_BINARY_PATH = os.path.join(
   PROJECT_ROOT, "llama.cpp-auto", "build", "bin", "llama-cli"
)

# Default model directory
DEFAULT_MODEL_DIR = os.path.expanduser("~/.vulkan_ilm/models")
DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.vulkan_ilm/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

# Vulkan Configuration
VULKAN_CONFIG = {
    "auto_detect": True,
    "default_gpu_layers": 100,
    "fallback_to_cpu": True,
    "vulkan_timeout": 30,
}

# Benchmark Configuration
BENCHMARK_CONFIG = {
    "default_prompt": "Hello, how are you today?",
    "default_iterations": 3,
    "max_tokens": 50,
    "temperature": 0.7,
}

# Server Configuration for Streaming
SERVER_CONFIG = {
    "default_host": "127.0.0.1",
    "default_port": 8080,
    "timeout": 30,
    "auto_port_scan": True,
}

# Model Management
MODEL_CONFIG = {
    "cache_dir": Path.home() / ".cache" / "vulkanilm" / "models",
    "default_quantization": "Q4_K_M",
    "auto_convert": True,
}

# Create directories
MODEL_CONFIG["cache_dir"].mkdir(parents=True, exist_ok=True)
Path(DEFAULT_MODEL_DIR).mkdir(parents=True, exist_ok=True)

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "log_file": Path.home() / ".cache" / "vulkanilm" / "logs" / "vulkanilm.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 3,
}

# Create log directory
LOG_CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
