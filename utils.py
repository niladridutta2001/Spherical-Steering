"""
Utility Functions for TruthfulQA Spherical Steering

This module provides utility functions for:
- Data loading and formatting for TruthfulQA
- Feature extraction using baukit
"""

import torch
import numpy as np
import random
from baukit import TraceDict
from datasets import load_dataset
from tqdm import tqdm
from truthfulqa_prompts import build_zero_shot_candidate


# ==================== Prompt Formatting ====================

def format_truthfulqa_prompt(question, choice, style="standard"):
    """
    Format a TruthfulQA question-choice pair into a prompt.
    
    Args:
        question: The question text
        choice: The answer choice
        style: "standard" for basic Q:A: format
    
    Returns:
        Formatted prompt string
    """
    if style == "standard":
        return f"Q: {question} A: {choice}"
    else:
        raise ValueError(f"Unknown style: {style}")


# ==================== Data Loading ====================

def get_truthfulqa_data(tokenizer, style="standard", seed=42):
    """
    Load TruthfulQA dataset and format for feature extraction.
    
    This function loads the TruthfulQA multiple choice dataset and creates
    (prompt, label, question_index) tuples for each answer choice.
    
    Args:
        tokenizer: HuggingFace tokenizer (unused but kept for compatibility)
        style: Prompt formatting style ("standard")
        seed: Random seed for reproducibility
    
    Returns:
        prompts: List[str] - Formatted prompts
        labels: np.array - Binary labels (1=truthful, 0=hallucination)
        q_indices: np.array - Question indices (for K-Fold grouping)
    """
    print(f"Loading TruthfulQA dataset (Style: {style}, MC2 targets)...")
    
    # Load dataset
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
    
    rng = random.Random(seed)
    all_questions = [item['question'] for item in dataset]
    
    all_prompts = []
    all_labels = []
    all_q_indices = []
    
    total_samples = 0
    
    for i, item in enumerate(dataset):
        question = item['question']
        targets = item['mc2_targets']
        choices = targets['choices']
        labels = targets['labels']
        
        for choice, label in zip(choices, labels):
            prompt = format_truthfulqa_prompt(question, choice, style)
            all_prompts.append(prompt)
            all_labels.append(label)
            all_q_indices.append(i)
            total_samples += 1
            
    print(f"Processed {len(dataset)} questions into {total_samples} samples.")
    return all_prompts, np.array(all_labels), np.array(all_q_indices)


def get_truthfulqa_candidates():
    """Return unformatted candidate rows for matched prompt construction."""
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")['validation']
    questions, choices, labels, q_indices, answer_indices = [], [], [], [], []
    for q_idx, item in enumerate(dataset):
        targets = item['mc2_targets']
        for answer_idx, (choice, label) in enumerate(zip(targets['choices'], targets['labels'])):
            questions.append(item['question']); choices.append(choice); labels.append(label)
            q_indices.append(q_idx); answer_indices.append(answer_idx)
    return (questions, choices, np.asarray(labels), np.asarray(q_indices),
            np.asarray(answer_indices))


# ==================== Feature Extraction ====================

def extract_layer_activations(model, tokenizer, prompts, layer_idx, device):
    """
    Extract last-token activations from a specific layer using baukit.
    
    This function processes prompts one at a time (batch_size=1) to ensure
    consistent behavior across different sequence lengths.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of prompt strings
        layer_idx: Layer index to extract from
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        activations: np.array of shape [N, hidden_dim]
    """
    layer_name = f"model.layers.{layer_idx}"
    activations = []
    
    print(f"Extracting activations from {layer_name}...")
    
    model.eval()
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting features"):
            # Tokenize single prompt
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            # Use baukit to trace layer outputs
            with TraceDict(model, [layer_name]) as ret:
                model(**inputs)
            
            # Get layer output
            layer_output = ret[layer_name].output
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]
            
            # Extract last token activation [1, Seq, Dim] -> [Dim]
            last_token_act = layer_output[0, -1, :].cpu().numpy()
            activations.append(last_token_act)
            
    return np.array(activations)


def extract_scored_activations(model, tokenizer, questions, choices, labels,
                               q_indices, answer_indices, layer_idx, device,
                               model_name=None, use_instruction=True,
                               feature_dtype="float16"):
    """Extract exactly the hidden positions whose logits score answer tokens."""
    layer_name = f"model.layers.{layer_idx}"
    activations, out_labels, out_q, out_answer = [], [], [], []
    token_offsets, answer_lengths, sample_weights = [], [], []
    np_dtype = np.float16 if feature_dtype == "float16" else np.float32
    model.eval()
    with torch.no_grad():
        rows = zip(questions, choices, labels, q_indices, answer_indices)
        for question, choice, label, q_idx, answer_idx in tqdm(
                rows, total=len(questions), desc="Extracting scored features"):
            built = build_zero_shot_candidate(
                tokenizer, question, choice, model_name, use_instruction, device)
            with TraceDict(model, [layer_name]) as ret:
                model(built["input_ids"])
            hidden = ret[layer_name].output
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            start, end, length = built["start_idx"], built["end_idx_exclusive"], built["answer_length"]
            selected = hidden[0, start:end, :].float().cpu().numpy().astype(np_dtype)
            if len(selected) != length:
                raise RuntimeError("scored activation count does not match answer token count")
            activations.extend(selected)
            out_labels.extend([label] * length)
            out_q.extend([q_idx] * length)
            out_answer.extend([answer_idx] * length)
            token_offsets.extend(range(length))
            answer_lengths.extend([length] * length)
            sample_weights.extend([1.0 / length] * length)
    return dict(
        activations=np.asarray(activations, dtype=np_dtype),
        labels=np.asarray(out_labels), q_indices=np.asarray(out_q),
        answer_indices=np.asarray(out_answer), token_offsets=np.asarray(token_offsets),
        answer_lengths=np.asarray(answer_lengths),
        sample_weights=np.asarray(sample_weights, dtype=np.float32))
