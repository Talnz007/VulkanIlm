# vulkan_ilm/installer.py

import subprocess
from pathlib import Path


def build_llama_cpp(dest="llama.cpp-auto", with_vulkan=True):
    repo = "https://github.com/ggerganov/llama.cpp.git"
    dest = Path(dest)
    if not dest.exists():
        print(f"Cloning llama.cpp to {dest}...")
        subprocess.check_call(["git", "clone", repo, str(dest)])
    else:
        print(f"Updating llama.cpp in {dest}...")
        subprocess.check_call(["git", "pull"], cwd=dest)

    build_cmd = ["cmake", "-B", "build"]
    if with_vulkan:
        build_cmd.append("-DGGML_VULKAN=1")

    subprocess.check_call(build_cmd, cwd=dest)
    subprocess.check_call(["cmake", "--build", "build", "--config", "Release"], cwd=dest)

    expected_bin = dest / "build" / "bin" / "llama-cli"
    assert expected_bin.exists(), "Build failed -- no llama-cli found."
    print("✅ llama.cpp built (Vulkan)", expected_bin)
    return expected_bin


# CLI usage for install
if __name__ == "__main__":
    build_llama_cpp()
