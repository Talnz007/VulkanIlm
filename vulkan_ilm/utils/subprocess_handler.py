"""
Subprocess handling utilities for VulkanIlm.
"""
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union


class LlamaProcess:
    """
    Manages communication with the llama.cpp process.
    """

    def __init__(
        self,
        binary_path: str,
        model_path: str,
        context_size: int = 2048,
        gpu_layers: int = 33,  # Default to all layers (33)
        seed: int = -1,
        threads: int = 12,  # Increased default threads
        verbose: bool = False,
    ):
        """
        Initialize a new LlamaProcess.

        Args:
            binary_path: Path to the llama.cpp binary
            model_path: Path to the GGUF model file
            context_size: Size of the context window in tokens
            gpu_layers: Number of layers to offload to GPU (33 for all)
            seed: Random seed for generation (-1 for random)
            threads: Number of CPU threads to use
            verbose: Whether to print verbose output
        """
        self.binary_path = binary_path
        self.model_path = model_path
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.seed = seed
        self.threads = threads
        self.verbose = verbose

        # Will be initialized when needed
        self.process = None

    def start(self) -> None:
        """Start the llama.cpp process."""
        if self.process is not None:
            return  # Process already started

        # Determine if using llama-cli (newer version) or main/llama
        binary_name = os.path.basename(self.binary_path)
        is_llama_cli = binary_name == "llama-cli"

        # Base command arguments
        cmd = [
            self.binary_path,
            "--model",
            self.model_path,
            "--ctx-size",
            str(self.context_size),
            "--threads",
            str(self.threads),
            "--seed",
            str(self.seed),
            "--color",
            "0",  # Disable color output for parsing
        ]

        # Add interactive mode based on binary type
        if is_llama_cli:
            cmd.append("--interactive")  # For llama-cli
        else:
            cmd.append("--interactive-first")  # For older binaries

        # Add GPU acceleration with correct flags
        if self.gpu_layers > 0:
            # Use -ngl or --n-gpu-layers depending on binary
            if is_llama_cli:
                cmd.extend(["--n-gpu-layers", str(self.gpu_layers)])
            else:
                cmd.extend(["-ngl", str(self.gpu_layers)])

        if self.verbose:
            cmd.append("--verbose")
            print(f"Starting process with command: {' '.join(cmd)}")

        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Add a small delay to allow the process to initialize
            time.sleep(2)

            # Read the initial prompt from llama.cpp (up to the first "> " prompt)
            initial_output = self._read_until_prompt()

            if self.verbose:
                print(f"Initial process output: {initial_output}")

        except Exception as e:
            print(f"Error starting process: {e}")
            if self.process:
                self.stop()
            raise

    def _read_until_prompt(self) -> str:
        """
        Read from stdout until the prompt character is found.

        Returns:
            The output read from the process
        """
        if self.process is None:
            raise RuntimeError("Process not started")

        output = []
        timeout = 30  # Timeout in seconds
        start_time = time.time()

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                if self.verbose:
                    print("Timeout waiting for prompt")
                break

            # Check if process has terminated
            if self.process.poll() is not None:
                if self.verbose:
                    print(f"Process terminated with return code: {self.process.returncode}")
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    print(f"Process stderr: {stderr_output}")
                raise RuntimeError(
                    f"Process terminated unexpectedly with return code: {self.process.returncode}"
                )

            # Try to read a line
            line = self.process.stdout.readline()
            if not line:
                # No more output, wait a bit and try again
                time.sleep(0.1)
                continue

            # Strip the line and add to output
            stripped_line = line.strip()
            output.append(stripped_line)

            if self.verbose:
                print(f"Read line: {stripped_line}")

            # Check for various prompt indicators
            if line.endswith("> ") or ">:" in line or line.strip() == ">":
                break

        return "\n".join(output)

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt
            params: Generation parameters

        Returns:
            Generated text
        """
        if self.process is None or self.process.poll() is not None:
            # Process has terminated or not started, start a new one
            self.process = None
            self.start()

        try:
            # For simpler testing, just send the prompt directly
            if self.verbose:
                print(f"Sending prompt: {prompt}")

            self.process.stdin.write(prompt + "\n")
            self.process.stdin.flush()

            # Read the response
            response = self._read_until_prompt()

            # Extract just the model's output
            if prompt in response:
                output_start = response.find(prompt) + len(prompt)
                output = response[output_start:].strip()
            else:
                output = response.strip()

            # Sometimes llama.cpp adds a trailing prompt character
            if output.endswith(">"):
                output = output[:-1].strip()

            return output

        except BrokenPipeError:
            # Handle broken pipe by restarting the process and trying again
            print("Broken pipe detected, restarting process...")
            self.stop()
            self.process = None
            self.start()

            # Try with a simpler approach
            try:
                self.process.stdin.write(prompt + "\n")
                self.process.stdin.flush()
                time.sleep(0.5)
                response = self._read_until_prompt()
                return response.strip()
            except Exception as e:
                print(f"Error during retry: {e}")
                raise

        except Exception as e:
            print(f"Error during generation: {e}")
            if self.process and self.process.poll() is None:
                # Process is still running, try to get stderr output
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    print(f"Process stderr: {stderr_output}")
            raise

    def stream_generate(self, prompt: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Stream text generation token by token.

        Args:
            prompt: The input prompt
            params: Generation parameters

        Yields:
            Tokens as they are generated
        """
        if self.process is None or self.process.poll() is not None:
            # Process has terminated or not started, start a new one
            self.process = None
            self.start()

        try:
            # Just send the prompt directly
            if self.verbose:
                print(f"Sending prompt for streaming: {prompt}")

            self.process.stdin.write(prompt + "\n")
            self.process.stdin.flush()

            # Read the response token by token
            output_started = False
            buffer = ""

            while True:
                # Check if process has terminated
                if self.process.poll() is not None:
                    print(f"Process terminated with return code: {self.process.returncode}")
                    break

                char = self.process.stdout.read(1)
                if not char:
                    time.sleep(0.1)  # Small pause to avoid CPU spinning
                    continue

                buffer += char

                # Wait until we see the end of the prompt/parameters
                if not output_started:
                    if prompt in buffer:
                        output_started = True
                        buffer = buffer[
                            buffer.find(prompt) + len(prompt) :
                        ]  # Keep only what's after the prompt
                    continue

                # Check for end of generation
                if char == ">" and buffer.endswith("> "):
                    break

                # Yield characters as they come
                yield char

        except Exception as e:
            print(f"Error during streaming: {e}")
            if self.process and self.process.poll() is None:
                # Process is still running, try to get stderr output
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    print(f"Process stderr: {stderr_output}")
            raise

    def stop(self) -> None:
        """Stop the llama.cpp process."""
        if self.process is not None:
            try:
                # Send SIGTERM signal
                self.process.terminate()

                # Wait for process to terminate (with timeout)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    self.process.kill()
                    self.process.wait()
            except:
                # Ignore errors during termination
                pass

            self.process = None
