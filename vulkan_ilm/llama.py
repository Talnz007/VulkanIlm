# vulkan_ilm/llama.py
"""
Core Llama class for Vulkan-accelerated LLM inference.
"""
import os
import platform
import time
from typing import Dict, Generator, List, Optional

from .simple_llama import SimpleLlama


class Llama:
    """
    Python interface for running LLMs on legacy GPUs using Vulkan.

    This class provides a simple API for loading and running inference on
    large language models using Vulkan acceleration via llama.cpp.
    """

    def __init__(
        self,
        model_path: str,
        binary_path: Optional[str] = None,
        context_size: int = 2048,
        gpu_layers: int = 100,  # Default to maximum GPU acceleration
        seed: int = -1,
        threads: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize a new Llama instance.

        Args:
            model_path: Path to the GGUF model file
            binary_path: Path to the llama.cpp binary (None for auto-detect)
            context_size: Size of the context window in tokens
            gpu_layers: Number of layers to offload to GPU (100 for max)
            seed: Random seed for generation (-1 for random)
            threads: Number of CPU threads to use (None for auto)
            verbose: Whether to print verbose output
        """
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.binary_path = binary_path or self._find_binary()
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.seed = seed
        self.threads = threads or os.cpu_count() or 4
        self.verbose = verbose

        # Validate model and binary
        self._validate_model()
        self._validate_binary()

        # Initialize the SimpleLlama implementation
        self.engine = SimpleLlama(
            model_path=self.model_path,
            binary_path=self.binary_path,
            gpu_layers=self.gpu_layers,
            context_size=self.context_size,
            threads=self.threads,
            seed=self.seed,
            verbose=self.verbose,
        )

        # Keep track of conversation history
        self.conversation_history = []

    def _find_binary(self) -> str:
        """
        Find the llama.cpp binary with Vulkan support.

        Returns:
            Path to the binary

        Raises:
            FileNotFoundError: If no binary is found
        """
        # Common locations to check
        possible_locations = []

        # Check in the project's external directory first
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_locations.extend(
            [
                os.path.join(project_root, "external", "llama.cpp", "build", "bin", "llama-cli"),
                os.path.join(project_root, "llama.cpp-auto", "build", "bin", "llama-cli"),
                os.path.join(project_root, "external", "llama.cpp", "build", "bin", "main"),
                os.path.join(project_root, "external", "llama.cpp", "build", "bin", "llama"),
            ]
        )

        # Platform-specific default locations
        system = platform.system().lower()
        if system == "linux":
            possible_locations.extend(
                [
                    os.path.expanduser("~/.local/bin/llama"),
                    "/usr/local/bin/llama",
                    "/usr/bin/llama",
                ]
            )
        elif system == "darwin":  # macOS
            possible_locations.extend(
                [
                    os.path.expanduser("~/llama.cpp/build/bin/main"),
                    os.path.expanduser("~/.local/bin/llama"),
                    "/usr/local/bin/llama",
                ]
            )
        elif system == "windows":
            possible_locations.extend(
                [
                    "llama.exe",
                    "main.exe",
                    "llama.cpp\\build\\bin\\Release\\main.exe",
                    "llama.cpp\\build\\bin\\Debug\\main.exe",
                    os.path.expanduser("~\\llama.cpp\\build\\bin\\Release\\main.exe"),
                ]
            )

        # Check each location
        for location in possible_locations:
            if os.path.exists(location) and os.access(location, os.X_OK):
                return os.path.abspath(location)

        # If we reach here, no binary was found
        raise FileNotFoundError(
            "Could not find llama.cpp binary. Please compile llama.cpp with Vulkan support "
            "and specify the path using binary_path parameter."
        )

    def _validate_model(self) -> None:
        """
        Validate that the model file exists and is accessible.

        Raises:
            FileNotFoundError: If the model file is not found
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Check that the file has a supported extension
        if not self.model_path.endswith((".gguf", ".bin")):
            raise ValueError(
                f"Unsupported model format: {self.model_path}. " "Please use a GGUF format model."
            )

    def _validate_binary(self) -> None:
        """
        Validate that the binary exists and is accessible.

        Raises:
            FileNotFoundError: If the binary is not found
            PermissionError: If the binary is not executable
        """
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

        if not os.access(self.binary_path, os.X_OK):
            raise PermissionError(f"Binary is not executable: {self.binary_path}")

    def ask(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        echo: bool = False,
    ) -> str:
        """
        Generate a response to the given prompt.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences that will stop generation
            echo: Whether to include the prompt in the response

        Returns:
            Generated text response
        """
        # Generate the response using the SimpleLlama implementation
        response = self.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Apply stop sequences if provided
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in response:
                    response = response.split(stop_seq, 1)[0]

        # Add to conversation history
        self.conversation_history.append((prompt, response))

        # Return the result
        if echo:
            return prompt + response
        else:
            return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> str:
        """
        Generate a chat response.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated chat response
        """
        # Generate the response using the SimpleLlama implementation
        response = self.engine.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # Add to conversation history if there are user messages
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            last_user_message = user_messages[-1]["content"]
            self.conversation_history.append((last_user_message, response))

        return response

    def stream_ask(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        echo: bool = False,
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token (simulated with character-by-character yield).

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences that will stop generation
            echo: Whether to include the prompt in the response

        Yields:
            Characters as they are "generated"
        """
        # Echo the prompt if requested
        if echo:
            for char in prompt:
                yield char

        # Generate the full response first (we can't stream with the SimpleLlama implementation)
        response = self.ask(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            echo=False,
        )

        # Yield the response character by character to simulate streaming
        for char in response:
            yield char
            # Add a small delay to simulate streaming
            time.sleep(0.01)

    def stream_ask_real(
        self,
        prompt: str,
        use_gpu: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 100,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream response tokens in real-time using llama-server"""
        try:
            from vulkan_ilm.streaming import LlamaServerStreaming
            from vulkan_ilm.vulkan.detector import VulkanDetector

            # Get Vulkan args if using GPU
            vulkan_args = []
            if use_gpu:
                detector = VulkanDetector()
                model_size_gb = 7.0  # Estimate or calculate from model file
                vulkan_args = detector.vulkan_args(model_size_gb)

            # Use streaming server
            with LlamaServerStreaming(self.model_path) as server:
                server.start_server(vulkan_args)
                for token in server.stream_completion(
                    prompt, max_tokens=max_tokens, temperature=temperature
                ):
                    yield token

        except Exception as e:
            # Fallback to simulated streaming
            print(f"Real streaming failed ({e}), using simulated streaming...")
            for char in self.stream_ask(
                prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
            ):
                yield char
