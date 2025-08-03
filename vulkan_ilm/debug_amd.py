# Create file: vulkan_ilm/debug_amd.py
import os
import subprocess
import time


def debug_llama_cli():
    # Path to your llama-cli binary
    binary_path = os.path.expanduser("~/Desktop/llama.cpp/build/bin/llama-cli")

    # Path to your model
    model_path = os.path.expanduser(
        "~/.vulkan_ilm/models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )

    # Try different Vulkan configurations
    vulkan_configs = [
        ["--vulkan-device", "0"],  # Try with explicit device ID
        ["--n-gpu-layers", "1"],  # Try with n-gpu-layers only
        [],  # Try without Vulkan (CPU only)
    ]

    for config in vulkan_configs:
        # Build the command
        cmd = [
            binary_path,
            "--model",
            model_path,
            "--ctx-size",
            "2048",
            "--threads",
            "12",
            "--seed",
            "-1",
            "--color",
            "0",
            "--interactive",
        ]

        # Add the current Vulkan configuration
        cmd.extend(config)

        print(f"\n\nTrying configuration: {' '.join(config) if config else 'CPU only'}")
        print(f"Full command: {' '.join(cmd)}")

        # Start the process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        try:
            # Give the process some time to initialize
            time.sleep(3)

            # Check if process is still running
            if process.poll() is not None:
                print(f"Process terminated with return code: {process.returncode}")
                stderr_output = process.stderr.read()
                if stderr_output:
                    print(f"Process stderr: {stderr_output}")
                continue  # Try next configuration

            # Read initial output
            print("Reading initial output...")
            output = []
            for _ in range(10):  # Read up to 10 lines
                if process.poll() is not None:
                    print(f"Process terminated with return code: {process.returncode}")
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        print(f"Process stderr: {stderr_output}")
                    break

                # Try to read a line with a timeout
                line = process.stdout.readline()
                if not line:
                    break

                output.append(line.strip())
                print(f"Line: {line.strip()}")

                if line.endswith("> "):
                    print("Found prompt character, ready to send commands")
                    break

            # If process is still running, we found a working configuration
            if process.poll() is None:
                print(f"SUCCESS with configuration: {' '.join(config) if config else 'CPU only'}")

                # Send a simple prompt
                prompt = "What is artificial intelligence? Answer in one sentence."
                print(f"Sending prompt: {prompt}")
                process.stdin.write(prompt + "\n")
                process.stdin.flush()

                # Read the response
                print("Reading response...")
                response_lines = []
                for _ in range(50):  # Read up to 50 lines or until prompt character
                    if process.poll() is not None:
                        print(f"Process terminated with return code: {process.returncode}")
                        break

                    line = process.stdout.readline()
                    if not line:
                        break

                    response_lines.append(line.strip())
                    print(f"Response line: {line.strip()}")

                    if line.endswith("> "):
                        print("Found end of response")
                        break

                print(f"Complete response: {' '.join(response_lines)}")

                # If we got here, we found a working configuration
                print("\nWorking configuration found!")
                return config

        except Exception as e:
            print(f"Error: {e}")

        finally:
            # Clean up
            if process.poll() is None:
                print("Terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    print("\nNo working configuration found.")
    return None


if __name__ == "__main__":
    working_config = debug_llama_cli()
    if working_config:
        print(
            f"\nRecommended configuration: {' '.join(working_config) if working_config else 'CPU only'}"
        )

        # Update the subprocess_handler.py file based on the working configuration
        if not working_config:  # CPU only
            print(
                "\nUpdate your subprocess_handler.py to remove the --vulkan flag and use CPU only."
            )
        else:
            print(
                f"\nUpdate your subprocess_handler.py to use the flags: {' '.join(working_config)}"
            )
