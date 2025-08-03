# vulkan_ilm/vulkan/detector.py
import json
import logging
import platform
import re
import subprocess


class VulkanDetector:
    def __init__(self):
        self.vulkan_available = False
        self.gpus = []
        self.best_gpu_index = 0
        self.detect_vulkan()

    def detect_vulkan(self):
        try:
            result = subprocess.run(
                ["vulkaninfo", "--json"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.vulkan_available = True
                self.parse_vulkaninfo_json(result.stdout)
        except Exception:
            # fallback simple string parsing
            try:
                result = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=10)
                self.vulkan_available = "Vulkan Instance Version" in result.stdout
            except Exception:
                self.vulkan_available = False

    def parse_vulkaninfo_json(self, vulkan_json):
        try:
            data = json.loads(vulkan_json)
            self.gpus = []
            for i, dev in enumerate(data.get("devices", [])):
                props = dev.get("properties", {})
                gpu = {
                    "index": i,
                    "name": props.get("deviceName", "Unknown GPU"),
                    "type": props.get("deviceType", 0),
                    "memory_mb": self._extract_mem(dev),
                }
                self.gpus.append(gpu)
            self.best_gpu_index = max(
                range(len(self.gpus)),
                key=lambda i: self.gpus[i]["memory_mb"] if "memory_mb" in self.gpus[i] else 0,
            )
        except Exception as e:
            logging.warning(f"Could not parse vulkaninfo: {e}")

    def _extract_mem(self, dev):
        try:
            heaps = dev.get("memoryProperties", {}).get("memoryHeaps", [])
            for heap in heaps:
                if heap.get("flags", 0) & 1:  # DEVICE_LOCAL
                    return int(heap.get("size", 0) // (1024 * 1024))
        except Exception:
            pass
        return 0

    def optimal_layers(self, model_size_gb):
        if not self.vulkan_available or not self.gpus:
            return 0
        mem = self.gpus[self.best_gpu_index]["memory_mb"]
        avail = 0.7 * (mem / 1024)
        if model_size_gb <= avail:
            return 999
        # Guess - fit as many as will go
        return max(1, int((avail / model_size_gb) * 35))

    def vulkan_args(self, model_size_gb):
        if not self.vulkan_available or not self.gpus:
            return []
        layers = self.optimal_layers(model_size_gb)
        args = ["-ngl", str(layers)]
        if len(self.gpus) > 1:
            args += ["-mg", str(self.best_gpu_index)]
        return args

    def print_gpu_info(self):
        if not self.vulkan_available:
            print("❌ No Vulkan GPUs Detected")
        for gpu in self.gpus:
            print(f"[{gpu['index']}] {gpu['name']} - {gpu['memory_mb']:,} MB")


# CLI usage (for vulkan-info command)
if __name__ == "__main__":
    v = VulkanDetector()
    v.print_gpu_info()
    if v.vulkan_available:
        print("Vulkan acceleration is AVAILABLE.")
    else:
        print("Vulkan is NOT available.")
