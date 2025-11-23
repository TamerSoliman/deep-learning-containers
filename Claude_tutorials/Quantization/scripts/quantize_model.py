#!/usr/bin/env python3
"""
Quantization Script for Large Language Models

Supports multiple quantization methods:
- INT8 (bitsandbytes)
- AWQ (4-bit)
- GPTQ (4-bit)
- NF4 (4-bit, supports QLoRA)

Usage:
    python quantize_model.py --model meta-llama/Llama-3-70b-hf --method awq --output ./llama-3-70b-awq
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_calibration_data(dataset_name="wikitext", n_samples=128):
    """Load calibration dataset for quantization"""
    logger.info(f"Loading calibration dataset: {dataset_name} ({n_samples} samples)")

    if dataset_name == "wikitext":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Filter out short samples
        data = data.filter(lambda x: len(x["text"]) > 512)
        texts = [data[i]["text"] for i in range(min(n_samples, len(data)))]

    elif dataset_name == "c4":
        data = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = []
        for i, sample in enumerate(data):
            if i >= n_samples:
                break
            if len(sample["text"]) > 512:
                texts.append(sample["text"])

    elif dataset_name == "alpaca":
        data = load_dataset("tatsu-lab/alpaca", split="train")
        texts = []
        for i in range(min(n_samples, len(data))):
            instruction = data[i]["instruction"]
            input_text = data[i]["input"]
            prompt = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                prompt += f"### Input:\n{input_text}\n\n"
            prompt += "### Response:\n"
            texts.append(prompt)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Loaded {len(texts)} calibration samples")
    return texts


def quantize_awq(model_path, output_dir, calibration_data, **kwargs):
    """Quantize model using AWQ (Activation-Aware Weight Quantization)"""
    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise ImportError("AWQ not installed. Run: pip install autoawq")

    logger.info("Loading model for AWQ quantization...")
    model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # AWQ quantization config
    quant_config = {
        "zero_point": kwargs.get("zero_point", True),
        "q_group_size": kwargs.get("group_size", 128),
        "w_bit": kwargs.get("bits", 4),
        "version": kwargs.get("version", "GEMM"),
        "alpha": kwargs.get("alpha", 0.5),
    }

    logger.info(f"AWQ config: {quant_config}")
    logger.info("Running quantization (this may take 15-30 minutes for 70B models)...")

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)

    # Save
    logger.info(f"Saving quantized model to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✓ AWQ quantization complete!")
    return output_dir


def quantize_gptq(model_path, output_dir, calibration_data, **kwargs):
    """Quantize model using GPTQ"""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        raise ImportError("GPTQ not installed. Run: pip install auto-gptq")

    logger.info("Loading model for GPTQ quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize calibration data
    logger.info("Tokenizing calibration data...")
    calibration_dataset = []
    for text in calibration_data:
        tokens = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
        calibration_dataset.append({"input_ids": tokens.input_ids[0]})

    # GPTQ quantization config
    quantize_config = BaseQuantizeConfig(
        bits=kwargs.get("bits", 4),
        group_size=kwargs.get("group_size", 128),
        desc_act=kwargs.get("desc_act", True),
        damp_percent=kwargs.get("damp_percent", 0.01),
        sym=kwargs.get("sym", False),
        true_sequential=kwargs.get("true_sequential", True),
    )

    logger.info(f"GPTQ config: {quantize_config.to_dict()}")

    # Load model
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        device_map="auto",
    )

    logger.info("Running quantization (this may take 1-2 hours for 70B models)...")
    model.quantize(calibration_dataset)

    # Save
    logger.info(f"Saving quantized model to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✓ GPTQ quantization complete!")
    return output_dir


def quantize_nf4(model_path, output_dir, **kwargs):
    """Quantize model using NF4 (bitsandbytes)"""
    logger.info("Loading model with NF4 quantization...")

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=kwargs.get("double_quant", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Save
    logger.info(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✓ NF4 quantization complete!")
    logger.info("Note: This model can be fine-tuned with QLoRA")
    return output_dir


def quantize_int8(model_path, output_dir, **kwargs):
    """Quantize model using INT8 (bitsandbytes)"""
    logger.info("Loading model with INT8 quantization...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto",
        llm_int8_threshold=kwargs.get("threshold", 6.0),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Save
    logger.info(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✓ INT8 quantization complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Quantize large language models")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--method", type=str, required=True, choices=["awq", "gptq", "nf4", "int8"],
                        help="Quantization method")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--calibration-dataset", type=str, default="wikitext",
                        choices=["wikitext", "c4", "alpaca"],
                        help="Calibration dataset (for AWQ/GPTQ)")
    parser.add_argument("--calibration-samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (4 or 8)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--alpha", type=float, default=0.5, help="AWQ alpha parameter")

    args = parser.parse_args()

    logger.info(f"Starting quantization: {args.method}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load calibration data if needed
    calibration_data = None
    if args.method in ["awq", "gptq"]:
        calibration_data = load_calibration_data(
            args.calibration_dataset,
            args.calibration_samples
        )

    # Run quantization
    if args.method == "awq":
        quantize_awq(
            args.model,
            args.output,
            calibration_data,
            bits=args.bits,
            group_size=args.group_size,
            alpha=args.alpha,
        )
    elif args.method == "gptq":
        quantize_gptq(
            args.model,
            args.output,
            calibration_data,
            bits=args.bits,
            group_size=args.group_size,
        )
    elif args.method == "nf4":
        quantize_nf4(args.model, args.output)
    elif args.method == "int8":
        quantize_int8(args.model, args.output)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Quantization Summary")
    logger.info("="*60)
    logger.info(f"Method: {args.method.upper()}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")

    if args.method in ["awq", "gptq"]:
        logger.info(f"Calibration: {args.calibration_dataset} ({args.calibration_samples} samples)")

    logger.info("\nNext steps:")
    logger.info("1. Test the quantized model:")
    logger.info(f"   python evaluate_quantized_model.py --model {args.output}")
    logger.info("2. Upload to S3:")
    logger.info(f"   tar -czf model.tar.gz -C {args.output} .")
    logger.info(f"   aws s3 cp model.tar.gz s3://your-bucket/models/")
    logger.info("3. Deploy to SageMaker")
    logger.info("="*60)


if __name__ == "__main__":
    main()
