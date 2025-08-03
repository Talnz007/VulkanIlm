# vulkan_ilm/benchmark.py
import time

from vulkan_ilm.llama import Llama


def benchmark(model_path, prompt="Hi", use_gpu=True, **gen_kwargs):
    gpu_layers = 100 if use_gpu else 0
    llm = Llama(model_path, gpu_layers=gpu_layers)
    start = time.time()
    resp = llm.ask(prompt, **gen_kwargs)
    elapsed = time.time() - start
    return elapsed, resp


def run_full_bench(model_path, prompt="Hi", **kwargs):
    print("Bench on CPU...")
    cpu_time, _ = benchmark(model_path, prompt, use_gpu=False, **kwargs)
    print("Bench on GPU...")
    gpu_time, _ = benchmark(model_path, prompt, use_gpu=True, **kwargs)
    print(f"CPU: {cpu_time:.2f}s | GPU: {gpu_time:.2f}s | Speedup: {cpu_time/gpu_time:.2f}x")
    return {"cpu_time": cpu_time, "gpu_time": gpu_time, "speedup": cpu_time / gpu_time}
