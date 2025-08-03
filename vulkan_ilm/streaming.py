# vulkan_ilm/streaming.py
import json
import logging
import subprocess
import time
from pathlib import Path

import requests


class LlamaServerStreaming:
    def __init__(self, model_path, server_bin=None, host="127.0.0.1", port=8080):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.proc = None

        # Find server binary (similar to your binary detection logic)
        if server_bin is None:
            server_bin = self._find_server_binary()
        self.server_bin = server_bin

    def _find_server_binary(self):
        """Find llama-server binary"""
        possible_paths = [
            "llama-server",
            "./llama-server",
            "./build/bin/llama-server",
            "./llama.cpp/build/bin/llama-server",
            "llama.cpp-auto/build/bin/llama-server",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).absolute())

        raise FileNotFoundError("llama-server binary not found. Run 'vulkanilm install' first.")

    def start_server(self, vulkan_args=None):
        """Start llama-server with Vulkan args"""
        cmd = [
            self.server_bin,
            "-m",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        if vulkan_args:
            cmd.extend(vulkan_args)

        print(f"Starting server: {' '.join(cmd)}")

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout=30):
        """Wait for server to start accepting requests"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Server ready!")
                    return
            except requests.RequestException:
                time.sleep(1)

        raise TimeoutError("Server failed to start within timeout")

    def stream_completion(self, prompt, max_tokens=100, temperature=0.7):
        """Stream tokens in real-time"""
        if not self.proc:
            raise RuntimeError("Server not started. Call start_server() first.")

        url = f"http://{self.host}:{self.port}/completion"
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            response = requests.post(url, json=data, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])
                            if "content" in json_data:
                                yield json_data["content"]
                        except json.JSONDecodeError:
                            continue

        except requests.RequestException as e:
            logging.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

    def stop_server(self):
        """Stop the server"""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            print("🛑 Server stopped")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()
