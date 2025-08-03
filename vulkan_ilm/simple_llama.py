# vulkan_ilm/simple_llama.py
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Union


class SimpleLlama:
    """
    Simple wrapper for llama.cpp with Vulkan acceleration.
    Uses non-interactive mode for direct text generation.
    """

    def __init__(
        self,
        model_path: str,
        binary_path: Optional[str] = None,
        gpu_layers: int = 25,  # Use 100 as it worked in manual testing
        context_size: int = 2048,
        threads: int = 4,
        seed: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize a new SimpleLlama instance.

        Args:
            model_path: Path to the GGUF model file
            binary_path: Path to the llama.cpp binary (auto-detected if None)
            gpu_layers: Number of layers to offload to GPU (100 for max)
            context_size: Size of the context window in tokens
            threads: Number of CPU threads to use
            seed: Random seed for generation (-1 for random)
            verbose: Whether to print verbose output
        """
        self.model_path = os.path.expanduser(model_path)
        self.binary_path = os.path.expanduser(binary_path or self._find_binary())
        self.gpu_layers = gpu_layers
        self.context_size = context_size
        self.threads = threads
        self.seed = seed
        self.verbose = verbose

        # Validate paths
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def _find_binary(self) -> str:
        """Find the llama.cpp binary with Vulkan support."""
        # Common locations to check
        possible_locations = [
            "~/Python_Projects/VulkanIlm/external/llama.cpp/build/bin/llama-cli",
            "~/Desktop/llama.cpp/build/bin/llama-cli",
            "~/Desktop/llama.cpp/build/bin/main",
            "~/Python_Projects/VulkanIlm/external/llama.cpp/build/bin/main",
        ]

        for loc in possible_locations:
            path = os.path.expanduser(loc)
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path

        raise FileNotFoundError(
            "Could not find llama-cli binary. Please specify the path manually."
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.5,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ):
        """
        Generate text using llama.cpp in non-interactive mode with real-time stdout capture.

        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Penalty for repeating tokens (not currently passed to llama.cpp)
            seed: Random seed (not currently implemented)

        Returns:
            Generated text as a string.
        """
        cmd = [
            self.binary_path,
            "-m",
            self.model_path,
            "-ngl",
            str(self.gpu_layers),  # Pass the GPU layers
            "-t",
            str(self.threads),  # Threads
            "-p",
            prompt,  # Prompt
            "-n",
            str(max_tokens),  # Number of tokens to predict
            "-e",  # Enable end-of-sequence token stopping
            "-no-cnv",
        ]

        # Add optional parameters
        if temperature != 0.8:
            cmd.extend(["--temp", str(temperature)])

        if top_p != 0.9:
            cmd.extend(["--top_p", str(top_p)])

        if top_k != 40:
            cmd.extend(["--top_k", str(top_k)])

        if self.verbose:
            print("Running command:", " ".join(cmd))

        try:
            # Launch process and capture output
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
            )
            stdout, stderr = process.communicate(timeout=600)  # adjust timeout as needed

            if self.verbose:
                print("Process finished. Parsing output...")

            if stderr and self.verbose:
                print("Stderr from llama.cpp:\n", stderr)

            return self._extract_generated_text(stdout, prompt)

        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("LLM generation timed out.")

        except Exception as e:
            raise RuntimeError(f"Error during generation: {e}")

    def _extract_generated_text(self, output: str, prompt: str) -> str:
        """Extract just the generated text from the command output."""
        # First, try to find the standardized output format with user/assistant tags
        if "<|user|>" in output and "<|assistant|>" in output:
            # Find the last assistant tag
            assistant_idx = output.rfind("<|assistant|>")
            if assistant_idx != -1:
                # Get everything after the assistant tag
                response = output[assistant_idx + len("<|assistant|>") :].strip()
                # Remove any trailing content after the response
                for marker in ["<|user|>", "llama_perf", ">"]:
                    if marker in response:
                        response = response[: response.find(marker)].strip()
                return response

        # If we can't find the assistant tag, try to find the prompt and extract everything after it
        if prompt in output:
            start_idx = output.find(prompt) + len(prompt)
            after_prompt = output[start_idx:].strip()

            # Look for assistant tag in what follows
            if "<|assistant|>" in after_prompt:
                after_tag = after_prompt[
                    after_prompt.find("<|assistant|>") + len("<|assistant|>") :
                ].strip()
                # Remove any trailing content after the response
                for marker in ["<|user|>", "llama_perf", ">"]:
                    if marker in after_tag:
                        after_tag = after_tag[: after_tag.find(marker)].strip()
                return after_tag
            else:
                # If no assistant tag, just return everything after the prompt
                # Remove any trailing debugging info
                for marker in ["llama_perf", ">"]:
                    if marker in after_prompt:
                        after_prompt = after_prompt[: after_prompt.find(marker)].strip()
                return after_prompt

        # Last resort: try to extract any content that looks like a response
        lines = output.split("\n")
        response_lines = []
        capture = False

        for line in lines:
            # Skip debug info lines
            if any(
                x in line
                for x in ["llama_model_loader", "ggml_vulkan", "print_info:", "llama_perf"]
            ):
                continue

            # If we see the prompt, start capturing after it
            if prompt in line:
                capture = True
                continue

            # If we see a line with assistant tag, start capturing
            if "<|assistant|>" in line:
                capture = True
                # Don't include the assistant tag
                line = line[line.find("<|assistant|>") + len("<|assistant|>") :].strip()

            # Stop capturing if we see user tag or performance info
            if any(x in line for x in ["<|user|>", "llama_perf", ">"]):
                break

            # If we're capturing and have meaningful content, add it
            if capture and line.strip():
                response_lines.append(line)

        if response_lines:
            return "\n".join(response_lines).strip()

        # If all else fails, return a generic message
        return "No response generated. Please try again with different parameters."

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate a chat response using a formatted prompt.

        Args:
            messages: List of chat messages, each with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Penalty for repeating tokens
            seed: Random seed for generation (overrides instance seed if provided)

        Returns:
            Generated chat response
        """
        # Extract system message if present
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)

        # Construct the chat prompt
        prompt_parts = []

        if system_message:
            prompt_parts.append(f"<|system|>\n{system_message}")

        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(f"<|user|>\n{msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"<|assistant|>\n{msg['content']}")

        # Add a final assistant prefix to prompt the model to respond
        prompt_parts.append("<|assistant|>")

        # Join the prompt parts
        prompt = "\n".join(prompt_parts)

        if self.verbose:
            print(f"Chat prompt:\n{prompt}")

        # Generate the response using the regular generate method
        response = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            seed=seed,
        )

        return response
