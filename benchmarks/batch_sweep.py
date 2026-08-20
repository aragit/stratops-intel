"""Benchmark script for vLLM extraction batching performance.

Tests batch sizes [1, 4, 8, 16, 32] with dummy prompts and measures
throughput (req/sec), latency p50/p99, GPU memory.
Outputs JSON report to benchmarks/reports/batch_sweep_{timestamp}.json
"""

import json
import os
import time
import datetime
from typing import Any, Dict, List

import structlog
import torch
from vllm import LLM, SamplingParams

logger = structlog.get_logger(__name__)

# Pre-defined simple schema for benchmarking
BENCHMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
    },
    "required": ["company_name"],
}


def build_benchmark_prompt(text: str) -> str:
    """Build a simple extraction prompt for benchmarking."""
    return f"""Extract structured information from the following text according to the JSON schema.
Only output valid JSON matching the schema. Do not include any explanation or extra text.

Schema:
{json.dumps(BENCHMARK_SCHEMA, indent=2)}

Text:
{text}

JSON Output:"""


def run_benchmarks() -> List[Dict[str, Any]]:
    """Run batch size sweep benchmarks.

    Tests batch sizes [1, 4, 8, 16, 32] with dummy prompts.
    Measures throughput (req/sec), latency p50/p99, GPU memory.

    Returns:
        List of result dicts with benchmark metrics per batch size.
    """
    model_id = "Qwen/Qwen2.5-7B-Instruct-AWQ"

    # Dummy prompts for benchmarking
    dummy_prompts = [
        "Apple Inc. is a technology company headquartered in Cupertino.",
        "Microsoft Corporation develops software and cloud services.",
        "Google parent company Alphabet focuses on search and advertising.",
        "Amazon.com dominates e-commerce and cloud infrastructure.",
        "Meta Platforms operates social media and advertising businesses.",
    ]

    batch_sizes = [1, 4, 8, 16, 32]
    results: List[Dict[str, Any]] = []

    for batch_size in batch_sizes:
        logger.info("running_benchmark_batch_size", batch_size=batch_size)

        # Select prompts for this batch size
        prompts = dummy_prompts[:batch_size]

        # Build all prompts with schema
        all_prompts = []
        for prompt_text in prompts:
            prompt = build_benchmark_prompt(prompt_text)
            all_prompts.append(prompt)

        # Initialize vLLM model
        llm = LLM(
            model=model_id,
            quantization="awq",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            enable_prefix_caching=True,
            trust_remote_code=True,
            dtype="half",
        )

        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=1024,
            stop=["<|endoftext|>"],
        )

        # Warmup
        logger.info("warming_up", batch_size=batch_size)
        llm.generate(all_prompts[:1], sampling_params)

        # Run benchmark
        num_iterations = 3
        latencies: List[float] = []

        for _ in range(num_iterations):
            start_time = time.time()
            outputs = llm.generate(all_prompts, sampling_params)
            end_time = time.time()

            batch_latency = (end_time - start_time) * 1000  # ms
            latencies.append(batch_latency)

            # Measure GPU memory
            gpu_memory = "0"
            if torch.cuda.is_available():
                gpu_memory = str(torch.cuda.max_memory_allocated())

        latencies_array = sorted(latencies)
        p50 = latencies_array[len(latencies_array) // 2]
        p99 = latencies_array[int(len(latencies_array) * 0.99)]

        throughput = (batch_size * num_iterations) / sum(latencies) * 1000

        result = {
            "batch_size": batch_size,
            "num_prompts": batch_size,
            "iterations": num_iterations,
            "latency_ms": {
                "p50": round(p50, 2),
                "p99": round(p99, 2),
                "mean": round(sum(latencies) / num_iterations, 2),
            },
            "throughput_req_sec": round(throughput, 2),
            "gpu_memory_bytes": int(gpu_memory),
            "gpu_memory_gb": round(int(gpu_memory) / (1024 ** 3), 2),
        }
        results.append(result)

        logger.info(
            "benchmark_complete",
            batch_size=batch_size,
            p50_ms=round(p50, 2),
            p99_ms=round(p99, 2),
            throughput=round(throughput, 2),
        )

        # Clean up
        del llm
        torch.cuda.empty_cache()

    return results


def main() -> None:
    """Run benchmarks and output JSON report."""
    results = run_benchmarks()

    # Generate timestamped report path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.getenv("BENCHMARK_REPORT_DIR", "/home/mobius/stratops-intel/benchmarks/reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"batch_sweep_{timestamp}.json")

    report = {
        "timestamp": timestamp,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "batch_sizes_tested": [1, 4, 8, 16, 32],
        "results": results,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()