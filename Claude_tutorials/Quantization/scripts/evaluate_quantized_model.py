#!/usr/bin/env python3
"""
Evaluation Script for Quantized Models

Evaluates quantized models on:
- Perplexity (WikiText-2)
- Generation quality (sample prompts)
- Inference speed (tokens/second)
- Memory usage

Usage:
    python evaluate_quantized_model.py --model ./llama-3-70b-awq --baseline meta-llama/Llama-3-70b-hf
"""

import argparse
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", n_samples=100):
    """Evaluate perplexity on WikiText-2"""
    logger.info(f"Evaluating perplexity on {dataset_name}...")

    # Load dataset
    if dataset_name == "wikitext":
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

    input_ids = encodings.input_ids.to(model.device)

    # Evaluate in chunks
    max_length = 2048
    stride = 512
    nlls = []

    logger.info(f"Processing {input_ids.size(1)} tokens...")

    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + max_length, input_ids.size(1))
        trg_len = end_loc - begin_loc
        input_chunk = input_ids[:, begin_loc:end_loc]

        with torch.no_grad():
            outputs = model(input_chunk, labels=input_chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        if (begin_loc // stride) % 10 == 0:
            logger.info(f"  Processed {begin_loc}/{input_ids.size(1)} tokens")

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    logger.info(f"✓ Perplexity: {ppl.item():.2f}")

    return ppl.item()


def evaluate_generation_quality(model, tokenizer, prompts=None):
    """Evaluate generation quality on sample prompts"""
    logger.info("Evaluating generation quality...")

    if prompts is None:
        prompts = [
            "Explain quantum computing in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
            "What is the capital of France?",
            "Solve: 2x + 5 = 13",
        ]

    results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"\nPrompt {i+1}: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0,  # Greedy decoding
                do_sample=False,
            )
        generation_time = time.time() - start_time

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()

        logger.info(f"Response: {response[:200]}...")
        logger.info(f"Time: {generation_time:.2f}s")

        results.append({
            "prompt": prompt,
            "response": response,
            "time": generation_time,
        })

    return results


def evaluate_speed(model, tokenizer, input_length=2048, output_length=100, n_runs=5):
    """Evaluate inference speed"""
    logger.info(f"\nEvaluating inference speed ({n_runs} runs)...")
    logger.info(f"Input length: {input_length} tokens, Output length: {output_length} tokens")

    # Create dummy input
    prompt = "The quick brown fox " * (input_length // 4)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_length, truncation=True).to(model.device)

    # Warmup
    logger.info("Warming up...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Benchmark
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        elapsed = time.time() - start_time
        times.append(elapsed)

        tokens_per_sec = output_length / elapsed
        logger.info(f"  Run {i+1}: {elapsed:.2f}s ({tokens_per_sec:.2f} tokens/sec)")

    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_tokens_per_sec = output_length / avg_time

    logger.info(f"\n✓ Average: {avg_time:.2f}s ± {std_time:.2f}s")
    logger.info(f"✓ Throughput: {avg_tokens_per_sec:.2f} tokens/sec")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "tokens_per_sec": avg_tokens_per_sec,
    }


def evaluate_memory(model):
    """Evaluate memory usage"""
    logger.info("\nEvaluating memory usage...")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB

        logger.info(f"✓ GPU Memory Allocated: {memory_allocated:.2f} GB")
        logger.info(f"✓ GPU Memory Reserved: {memory_reserved:.2f} GB")

        return {
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
        }
    else:
        logger.info("CUDA not available, skipping memory evaluation")
        return {}


def compare_with_baseline(quantized_results, baseline_results):
    """Compare quantized model with baseline"""
    logger.info("\n" + "="*80)
    logger.info("COMPARISON WITH BASELINE")
    logger.info("="*80)

    # Perplexity
    if "perplexity" in quantized_results and "perplexity" in baseline_results:
        ppl_quant = quantized_results["perplexity"]
        ppl_base = baseline_results["perplexity"]
        ppl_delta = ((ppl_quant - ppl_base) / ppl_base) * 100

        logger.info(f"\nPerplexity:")
        logger.info(f"  Baseline:   {ppl_base:.2f}")
        logger.info(f"  Quantized:  {ppl_quant:.2f}")
        logger.info(f"  Change:     {ppl_delta:+.1f}%")

    # Speed
    if "speed" in quantized_results and "speed" in baseline_results:
        speed_quant = quantized_results["speed"]["tokens_per_sec"]
        speed_base = baseline_results["speed"]["tokens_per_sec"]
        speedup = speed_quant / speed_base

        logger.info(f"\nInference Speed:")
        logger.info(f"  Baseline:   {speed_base:.2f} tokens/sec")
        logger.info(f"  Quantized:  {speed_quant:.2f} tokens/sec")
        logger.info(f"  Speedup:    {speedup:.2f}x")

    # Memory
    if "memory" in quantized_results and "memory" in baseline_results:
        mem_quant = quantized_results["memory"].get("memory_allocated_gb", 0)
        mem_base = baseline_results["memory"].get("memory_allocated_gb", 0)

        if mem_base > 0:
            mem_reduction = ((mem_base - mem_quant) / mem_base) * 100

            logger.info(f"\nMemory Usage:")
            logger.info(f"  Baseline:   {mem_base:.2f} GB")
            logger.info(f"  Quantized:  {mem_quant:.2f} GB")
            logger.info(f"  Reduction:  {mem_reduction:.1f}%")

    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized models")
    parser.add_argument("--model", type=str, required=True, help="Path to quantized model")
    parser.add_argument("--baseline", type=str, help="Baseline model for comparison (optional)")
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation quality")
    parser.add_argument("--skip-speed", action="store_true", help="Skip speed benchmark")
    parser.add_argument("--n-speed-runs", type=int, default=5, help="Number of speed benchmark runs")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("QUANTIZED MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")

    # Load quantized model
    logger.info("\nLoading quantized model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    quantized_results = {}

    # Evaluate perplexity
    if not args.skip_perplexity:
        ppl = evaluate_perplexity(model, tokenizer)
        quantized_results["perplexity"] = ppl

    # Evaluate generation quality
    if not args.skip_generation:
        gen_results = evaluate_generation_quality(model, tokenizer)
        quantized_results["generation"] = gen_results

    # Evaluate speed
    if not args.skip_speed:
        speed_results = evaluate_speed(model, tokenizer, n_runs=args.n_speed_runs)
        quantized_results["speed"] = speed_results

    # Evaluate memory
    memory_results = evaluate_memory(model)
    quantized_results["memory"] = memory_results

    # Compare with baseline if provided
    if args.baseline:
        logger.info(f"\nLoading baseline model: {args.baseline}...")
        baseline_model = AutoModelForCausalLM.from_pretrained(
            args.baseline,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline)

        baseline_results = {}

        if not args.skip_perplexity:
            baseline_results["perplexity"] = evaluate_perplexity(baseline_model, baseline_tokenizer)

        if not args.skip_speed:
            baseline_results["speed"] = evaluate_speed(baseline_model, baseline_tokenizer, n_runs=args.n_speed_runs)

        baseline_results["memory"] = evaluate_memory(baseline_model)

        compare_with_baseline(quantized_results, baseline_results)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")

    if "perplexity" in quantized_results:
        logger.info(f"Perplexity: {quantized_results['perplexity']:.2f}")

    if "speed" in quantized_results:
        logger.info(f"Speed: {quantized_results['speed']['tokens_per_sec']:.2f} tokens/sec")

    if "memory" in quantized_results and "memory_allocated_gb" in quantized_results["memory"]:
        logger.info(f"Memory: {quantized_results['memory']['memory_allocated_gb']:.2f} GB")

    logger.info("="*80)


if __name__ == "__main__":
    main()
