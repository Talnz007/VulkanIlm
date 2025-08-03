"""
Command-line interface for VulkanIlm.
"""
import os
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from vulkan_ilm.benchmark import run_full_bench
from vulkan_ilm.installer import build_llama_cpp
from vulkan_ilm.vulkan.detector import VulkanDetector

from .llama import Llama
from .models.manager import ModelManager

# Create a rich console for pretty output
console = Console()


@click.group()
@click.version_option()
def cli():
    """VulkanIlm - Run LLMs on legacy GPUs using Vulkan."""
    pass


@cli.command()
def vulkan_info():
    """Show detected Vulkan GPUs and info."""
    v = VulkanDetector()
    v.print_gpu_info()
    print("Vulkan available?", v.vulkan_available)


@cli.command()
def install():
    """Auto-install llama.cpp with Vulkan."""
    build_llama_cpp()


@cli.command()
@click.argument("model_path")
@click.option("--binary-path", "-b", help="Path to the llama.cpp binary")
@click.option("--prompt", "-p", help="Text prompt to process")
@click.option("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
@click.option("--max-tokens", "-m", type=int, default=512, help="Maximum tokens to generate")
@click.option("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
@click.option("--top-k", type=int, default=40, help="Top-k sampling parameter")
@click.option("--stream/--no-stream", default=True, help="Stream tokens as they're generated")
@click.option("--gpu/--cpu", default=True, help="Use GPU or CPU")
def ask(
    model_path: str,
    binary_path: Optional[str],
    prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    stream: bool,
    gpu: bool,
):
    """Generate text from a prompt using the specified model."""
    try:
        # Load the model
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Loading model..."),
            transient=True,
        ) as progress:
            progress.add_task("load", total=None)
            model = Llama(model_path, binary_path=binary_path)

        # If no prompt provided, enter interactive mode
        if not prompt:
            console.print("[bold green]VulkanIlm Interactive Mode[/bold green]")
            console.print(
                "Type your prompts below. Use [bold]Ctrl+C[/bold] or type [bold]exit[/bold] to quit.\n"
            )

            while True:
                try:
                    user_prompt = console.input("[bold cyan]You:[/bold cyan] ")
                    if user_prompt.lower() in ("exit", "quit"):
                        break

                    # Stream the response for a better UX
                    console.print("[bold green]Llama:[/bold green] ", end="")

                    if stream:
                        # Use the new streaming method
                        try:
                            for token in model.stream_ask(
                                user_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                top_k=top_k,
                            ):
                                print(token, end="", flush=True)
                        except Exception as e:
                            console.print(f"[red]Streaming error: {e}[/red]")
                            # Fallback to regular ask
                            response = model.ask(
                                user_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                top_k=top_k,
                            )
                            console.print(response)
                        console.print()
                    else:
                        # Generate the full response at once with a spinner
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[bold blue]Thinking..."),
                            transient=True,
                        ) as progress:
                            progress.add_task("generate", total=None)
                            response = model.ask(
                                user_prompt,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                top_k=top_k,
                            )
                        console.print(response)

                    console.print()

                except KeyboardInterrupt:
                    console.print("\n[bold yellow]Exiting...[/bold yellow]")
                    break
        else:
            # Process a single prompt
            if stream:
                # Stream tokens as they're generated
                console.print("[bold green]Response:[/bold green] ", end="")
                try:
                    for token in model.stream_ask(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                    ):
                        print(token, end="", flush=True)
                except Exception as e:
                    console.print(f"[red]Streaming error: {e}[/red]")
                    # Fallback to regular ask
                    response = model.ask(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    console.print(response)
                console.print()
            else:
                # Generate the full response at once with a spinner
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Generating..."),
                    BarColumn(),
                    TimeElapsedColumn(),
                ) as progress:
                    task = progress.add_task("generate", total=None)
                    response = model.ask(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    progress.update(task, completed=100)

                # Pretty-print the response
                try:
                    # Try to render as markdown for a nicer display
                    console.print(Markdown(response))
                except Exception as e:
                    # Fall back to plain text if markdown parsing fails
                    console.print(response)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


# Add streaming command
@cli.command()
@click.argument("model_path")
@click.argument("prompt", type=str)
@click.option("--binary-path", "-b", help="Path to the llama.cpp binary")
@click.option("--gpu/--cpu", default=True, help="Use GPU or CPU")
@click.option("--max-tokens", default=100, help="Maximum tokens to generate")
def stream(model_path, prompt, binary_path, gpu, max_tokens):
    """Stream response tokens in real-time using llama-server."""
    try:
        console.print("🔥 Starting real-time streaming...")

        # Import here to avoid circular imports
        from vulkan_ilm.streaming import LlamaServerStreaming
        from vulkan_ilm.vulkan.detector import VulkanDetector

        # Get Vulkan args if using GPU
        vulkan_args = []
        if gpu:
            detector = VulkanDetector()
            model_size_gb = 7.0  # Estimate
            vulkan_args = detector.vulkan_args(model_size_gb)

        # Use streaming server
        with LlamaServerStreaming(model_path) as server:
            server.start_server(vulkan_args)
            console.print("[bold green]Response:[/bold green] ", end="")

            for token in server.stream_completion(prompt, max_tokens=max_tokens):
                print(token, end="", flush=True)

            console.print("\n" + "-" * 30)

    except KeyboardInterrupt:
        console.print("\n\n⏹️ Stream interrupted")
    except Exception as e:
        console.print(f"\n❌ Streaming error: {e}")


@cli.command()
@click.argument("query", required=False, default="")
@click.option("--limit", "-l", type=int, default=20, help="Maximum number of results")
def search(query: str, limit: int):
    """Search for compatible models on HuggingFace."""
    try:
        manager = ModelManager()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Searching for models..."),
            transient=True,
        ) as progress:
            progress.add_task("search", total=None)
            models = manager.search_models(query, limit=limit)

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return

        console.print(f"[bold green]Found {len(models)} models:[/bold green]")

        for i, model in enumerate(models):
            # Print model ID and stats
            console.print(f"[bold cyan]{i + 1}.[/bold cyan] [blue]{model['id']}[/blue]")

            # Print additional details if available
            if "downloads" in model:
                console.print(f"   [grey]Downloads:[/grey] {model['downloads']:,}")

            if "last_modified" in model:
                console.print(f"   [grey]Last updated:[/grey] {model['last_modified']}")

            if "tags" in model and model["tags"]:
                tags = ", ".join(model["tags"])
                console.print(f"   [grey]Tags:[/grey] {tags}")

            if "files" in model and model["files"]:
                console.print("   [grey]Available files:[/grey]")
                for file in model["files"]:
                    console.print(f"   - {file}")

            # Add a separator between models
            if i < len(models) - 1:
                console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option("--filename", "-f", help="Specific filename to download")
def download(model_id: str, filename: Optional[str]):
    """Download a model from HuggingFace."""
    try:
        manager = ModelManager()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Downloading model..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("download", total=None)
            path = manager.download_model(model_id, filename=filename)
            progress.update(task, completed=100)

        console.print(f"[bold green]Model downloaded to:[/bold green] {path}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
def list():
    """List locally available models."""
    try:
        manager = ModelManager()
        models = manager.list_local_models()

        if not models:
            console.print("[yellow]No local models found.[/yellow]")
            return

        console.print(f"[bold green]Found {len(models)} local models:[/bold green]")

        for i, model_path in enumerate(models):
            model_name = os.path.basename(model_path)
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB

            console.print(f"[bold cyan]{i + 1}.[/bold cyan] [blue]{model_name}[/blue]")
            console.print(f"   [grey]Path:[/grey] {model_path}")
            console.print(f"   [grey]Size:[/grey] {model_size:.1f} MB")

            # Add a separator between models
            if i < len(models) - 1:
                console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("prompt", type=str, default="Hello world!")
def benchmark(prompt):
    """Benchmark CPU vs GPU performance."""
    try:
        console.print("🏃 Running benchmark...")
        # Use default model from config
        from vulkan_ilm.config import DEFAULT_MODEL_PATH
        from vulkan_ilm.llama import Llama

        results = run_full_bench(DEFAULT_MODEL_PATH, prompt)
        console.print(f"[bold green]Benchmark Results:[/bold green]")
        console.print(f"CPU Time: {results.get('cpu_time', 'N/A'):.2f}s")
        console.print(f"GPU Time: {results.get('gpu_time', 'N/A'):.2f}s")
        console.print(f"Speedup: {results.get('speedup', 'N/A'):.2f}x")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
