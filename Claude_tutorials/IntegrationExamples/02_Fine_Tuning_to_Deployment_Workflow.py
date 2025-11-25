#!/usr/bin/env python3
"""
Complete Fine-Tuning to Deployment Workflow

This example demonstrates the full ML lifecycle:
1. Fine-tune Llama 3 8B on custom data (AWS Trainium/GPU)
2. Evaluate fine-tuned model
3. Compile for inference (optional: Neuron)
4. Deploy to SageMaker
5. A/B test against base model
6. Promote to production

Architecture:
    Training Data → SageMaker Training (HF DLC) → Fine-tuned Model
                                                       ↓
                                              Model Evaluation
                                                       ↓
                                          Compile for Neuron (optional)
                                                       ↓
                    A/B Test Deployment → Production Endpoint
"""

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from sagemaker import Model
import json
import time
from typing import Dict, List

# ============================================================================
# STEP 1: Fine-Tuning on SageMaker
# ============================================================================

def fine_tune_model_gpu(
    training_data_s3: str,
    base_model: str = "meta-llama/Llama-3-8b",
    output_s3: str = "s3://your-bucket/models/llama-3-8b-finetuned",
):
    """
    Fine-tune model on GPU using HuggingFace DLC

    Args:
        training_data_s3: S3 path to training data (JSONL format)
        base_model: Base model ID
        output_s3: S3 path for output model
    """
    role = sagemaker.get_execution_role()

    # Training script (saved separately as train.py)
    train_script = """
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    # Load dataset
    dataset = load_dataset('json', data_files='/opt/ml/input/data/training/train.jsonl')

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        os.environ['BASE_MODEL'],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ['BASE_MODEL'])
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    # Tokenize dataset
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=2048)

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

if __name__ == "__main__":
    main()
"""

    # HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='./scripts',  # Contains train.py
        instance_type='ml.p4d.24xlarge',  # 8x A100
        instance_count=1,
        role=role,
        transformers_version='4.56',
        pytorch_version='2.8',
        py_version='py312',
        hyperparameters={
            'BASE_MODEL': base_model,
        },
        environment={
            'HF_TOKEN': 'hf_...',  # Your HuggingFace token
        },
        output_path=output_s3,
    )

    # Start training
    huggingface_estimator.fit({'training': training_data_s3})

    return huggingface_estimator.model_data


def fine_tune_model_trainium(
    training_data_s3: str,
    base_model: str = "meta-llama/Llama-3-8b",
    output_s3: str = "s3://your-bucket/models/llama-3-8b-finetuned-neuron",
):
    """
    Fine-tune on AWS Trainium (30-50% cost savings)
    """
    role = sagemaker.get_execution_role()

    huggingface_estimator = HuggingFace(
        entry_point='train_neuron.py',
        source_dir='./scripts',
        instance_type='ml.trn1.32xlarge',  # 16 NeuronCores
        instance_count=1,
        role=role,
        image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training-neuronx:2.8-...',
        hyperparameters={
            'BASE_MODEL': base_model,
            'NEURON_NUM_CORES': 16,
        },
        output_path=output_s3,
    )

    huggingface_estimator.fit({'training': training_data_s3})

    return huggingface_estimator.model_data


# ============================================================================
# STEP 2: Model Evaluation
# ============================================================================

def evaluate_model(model_s3_path: str, test_data_s3: str) -> Dict:
    """
    Evaluate fine-tuned model on test set

    Returns metrics: perplexity, accuracy, etc.
    """
    # Create processing job to evaluate model
    from sagemaker.processing import ScriptProcessor

    role = sagemaker.get_execution_role()

    processor = ScriptProcessor(
        role=role,
        image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.8-gpu-py312',
        instance_count=1,
        instance_type='ml.g5.xlarge',
        command=['python3'],
    )

    # Evaluation script
    eval_script = """
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# Load model
model = AutoModelForCausalLM.from_pretrained('/opt/ml/processing/input/model')
tokenizer = AutoTokenizer.from_pretrained('/opt/ml/processing/input/model')

# Load test data
test_data = load_dataset('json', data_files='/opt/ml/processing/input/data/test.jsonl')

# Evaluate perplexity
perplexities = []
for example in test_data['train']:
    inputs = tokenizer(example['text'], return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        perplexities.append(torch.exp(outputs.loss).item())

metrics = {
    'perplexity': np.mean(perplexities),
    'perplexity_std': np.std(perplexities),
}

# Save metrics
with open('/opt/ml/processing/output/metrics.json', 'w') as f:
    json.dump(metrics, f)
"""

    processor.run(
        code='evaluate.py',
        inputs=[
            {'source': model_s3_path, 'destination': '/opt/ml/processing/input/model'},
            {'source': test_data_s3, 'destination': '/opt/ml/processing/input/data'},
        ],
        outputs=[{'source': '/opt/ml/processing/output', 'destination': 's3://your-bucket/eval-results'}],
    )

    # Read metrics from S3
    # ... (implementation)

    return {"perplexity": 5.8, "accuracy": 0.85}  # Example


# ============================================================================
# STEP 3: Deploy and A/B Test
# ============================================================================

def deploy_with_ab_testing(
    base_model_s3: str,
    finetuned_model_s3: str,
    traffic_split: tuple = (0.5, 0.5),
):
    """
    Deploy both models with A/B testing

    Args:
        base_model_s3: S3 path to base model
        finetuned_model_s3: S3 path to fine-tuned model
        traffic_split: (base_traffic, finetuned_traffic) e.g., (0.5, 0.5)
    """
    role = sagemaker.get_execution_role()

    # Create models
    base_model = Model(
        image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.11.2-gpu-sagemaker',
        model_data=base_model_s3,
        role=role,
        env={'SM_VLLM_MODEL': 'meta-llama/Llama-3-8b'},
    )

    finetuned_model = Model(
        image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.11.2-gpu-sagemaker',
        model_data=finetuned_model_s3,
        role=role,
    )

    # Create endpoint config with traffic split
    from sagemaker.model import Model
    from sagemaker.session import Session

    endpoint_config_name = f"ab-test-config-{int(time.time())}"
    endpoint_name = f"ab-test-endpoint-{int(time.time())}"

    session = Session()

    session.create_endpoint_config(
        name=endpoint_config_name,
        model_name=base_model.name,
        initial_instance_count=1,
        instance_type='ml.g5.2xlarge',
        variant_name='BaseModel',
        initial_weight=traffic_split[0],
    )

    # Add second variant
    session.create_endpoint_config(
        name=endpoint_config_name,
        model_name=finetuned_model.name,
        initial_instance_count=1,
        instance_type='ml.g5.2xlarge',
        variant_name='FinetunedModel',
        initial_weight=traffic_split[1],
    )

    # Create endpoint
    session.create_endpoint(
        endpoint_name=endpoint_name,
        config_name=endpoint_config_name,
        wait=True,
    )

    return endpoint_name


# ============================================================================
# STEP 4: Monitor A/B Test and Promote Winner
# ============================================================================

def monitor_ab_test(endpoint_name: str, duration_hours: int = 24):
    """
    Monitor A/B test metrics

    Compares:
    - Model latency
    - Model quality (via human feedback or automated metrics)
    - Cost per request
    """
    import boto3

    cloudwatch = boto3.client('cloudwatch')

    # Collect metrics for both variants
    metrics = {}

    for variant in ['BaseModel', 'FinetunedModel']:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='ModelLatency',
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name},
                {'Name': 'VariantName', 'Value': variant},
            ],
            StartTime=time.time() - duration_hours * 3600,
            EndTime=time.time(),
            Period=3600,
            Statistics=['Average', 'Maximum', 'SampleCount'],
        )

        metrics[variant] = {
            'avg_latency': np.mean([dp['Average'] for dp in response['Datapoints']]),
            'max_latency': max([dp['Maximum'] for dp in response['Datapoints']]),
            'request_count': sum([dp['SampleCount'] for dp in response['Datapoints']]),
        }

    print(f"A/B Test Results ({duration_hours} hours):")
    print(f"Base Model: {metrics['BaseModel']}")
    print(f"Fine-tuned Model: {metrics['FinetunedModel']}")

    # Decide winner (example logic)
    if metrics['FinetunedModel']['avg_latency'] < metrics['BaseModel']['avg_latency']:
        print("✓ Fine-tuned model is faster - promoting to production!")
        return 'FinetunedModel'
    else:
        print("Base model is still better - keeping as is")
        return 'BaseModel'


def promote_to_production(endpoint_name: str, winning_variant: str):
    """
    Shift all traffic to winning variant
    """
    session = sagemaker.Session()

    session.update_endpoint_weights_and_capacities(
        endpoint_name=endpoint_name,
        desired_weights_and_capacities=[
            {'VariantName': 'BaseModel', 'DesiredWeight': 0.0 if winning_variant != 'BaseModel' else 1.0},
            {'VariantName': 'FinetunedModel', 'DesiredWeight': 1.0 if winning_variant == 'FinetunedModel' else 0.0},
        ]
    )

    print(f"✓ Shifted 100% traffic to {winning_variant}")


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================

def main():
    """Complete fine-tuning to deployment workflow"""

    print("="*80)
    print("STEP 1: Fine-Tuning")
    print("="*80)

    # Option A: Fine-tune on GPU (faster, more expensive)
    model_s3 = fine_tune_model_gpu(
        training_data_s3='s3://your-bucket/data/train.jsonl',
        base_model='meta-llama/Llama-3-8b',
    )

    # Option B: Fine-tune on Trainium (slower, 30-50% cheaper)
    # model_s3 = fine_tune_model_trainium(...)

    print(f"✓ Model saved to: {model_s3}")

    print("\n" + "="*80)
    print("STEP 2: Evaluation")
    print("="*80)

    metrics = evaluate_model(
        model_s3_path=model_s3,
        test_data_s3='s3://your-bucket/data/test.jsonl',
    )

    print(f"Metrics: {metrics}")

    if metrics['perplexity'] > 10.0:
        print("⚠️ Warning: High perplexity - model may need more training")
        return

    print("\n" + "="*80)
    print("STEP 3: A/B Test Deployment")
    print("="*80)

    endpoint = deploy_with_ab_testing(
        base_model_s3='s3://your-bucket/models/base-model',
        finetuned_model_s3=model_s3,
        traffic_split=(0.5, 0.5),  # 50-50 split
    )

    print(f"✓ Deployed A/B test endpoint: {endpoint}")

    print("\n" + "="*80)
    print("STEP 4: Monitor and Promote")
    print("="*80)

    # Monitor for 24 hours
    winner = monitor_ab_test(endpoint, duration_hours=24)

    # Promote winner
    promote_to_production(endpoint, winner)

    print("\n✓ Workflow complete!")


if __name__ == "__main__":
    main()
