"""
Step 1 (Generic): Extract Hidden State Features for MC Tasks

Usage:
    python get_activations_generic.py --model_name llama3.1-8B --dataset copa --layer 14
    python get_activations_generic.py --model_name llama3.1-8B --dataset mmlu_global --layer 14
    python get_activations_generic.py --model_name llama3.1-8B --dataset storycloze --layer 14
    python get_activations_generic.py --model_name llama3.1-8B --dataset winogrande --layer 14 --num_samples 2000
    python get_activations_generic.py --model_name llama3.1-8B --dataset boolq --layer 14

Output:
    Saves a .npz file containing:
    - activations: [N, hidden_dim] array of last-token hidden states
    - labels: [N] binary labels (1=correct, 0=incorrect)
    - q_indices: [N] question indices for K-Fold splitting
"""

import argparse
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_generic import (HF_NAMES, set_seed, get_dataset_data, get_layer_activations,
                           get_winogrande_scored_data, get_mmlu_scored_data,
                           get_copa_scored_data, get_scored_activations)
from utils_generic import get_boolq_scored_data, get_storycloze_scored_data


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract features for MC tasks")
    parser.add_argument('--model_name', type=str, default='llama3.1-8B-Instruct',
                        help=f"Model name or path. Shortcuts: {list(HF_NAMES.keys())}")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['copa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'])
    parser.add_argument('--layer', type=int, default=14, help="Layer index to extract")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_samples', type=int, default=None, help="Limit number of questions")
    parser.add_argument('--save_dir', type=str, default='./features_generic')
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Local model directory (overrides model_name)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--activation-positions', choices=['last', 'scored'], default='last')
    parser.add_argument('--feature-dtype', choices=['float16', 'float32'], default='float32')

    args = parser.parse_args()
    set_seed(args.seed)

    # 1. Load Model
    model_path = args.model_dir if args.model_dir else HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Load Data
    if args.activation_positions == 'scored':
        if args.split != 'train' or args.dataset not in (
                'winogrande', 'mmlu_global', 'copa', 'boolq', 'storycloze'):
            raise ValueError(
                'scored extraction supports WinoGrande, MMLU, COPA, BoolQ, or StoryCloze train only')
        if args.dataset == 'winogrande':
            n = args.num_samples if args.num_samples is not None else 1000
            prompts, candidates, labels, q_indices, answer_indices = \
                get_winogrande_scored_data(n, args.seed)
            category_indices = None
        elif args.dataset == 'mmlu_global':
            n = args.num_samples if args.num_samples is not None else 500
            (prompts, candidates, labels, q_indices, answer_indices,
             category_indices) = get_mmlu_scored_data(tokenizer, args.seed, n)
        elif args.dataset == 'copa':
            if args.num_samples not in (None, 400):
                raise ValueError('COPA scored extraction uses all 400 training questions')
            prompts, candidates, labels, q_indices, answer_indices = \
                get_copa_scored_data(args.seed)
            category_indices = None
        elif args.dataset == 'boolq':
            n = args.num_samples if args.num_samples is not None else 1000
            prompts, candidates, labels, q_indices, answer_indices = \
                get_boolq_scored_data(n, args.seed)
            category_indices = None
        else:
            if args.num_samples is not None:
                raise ValueError('StoryCloze scored extraction uses the complete train split')
            prompts, candidates, labels, q_indices, answer_indices = \
                get_storycloze_scored_data()
            category_indices = None
    else:
        prompts, labels, q_indices = get_dataset_data(
            args.dataset, split=args.split, num_samples=args.num_samples,
            seed=args.seed)

    print(f"\n=== Data Statistics ===")
    print(f"Total samples: {len(prompts)}")
    print(f"Correct (label=1): {sum(labels)}, Incorrect (label=0): {len(labels) - sum(labels)}")
    print(f"Example prompt: {prompts[0][:200]}...")

    # 3. Extract Features
    if args.activation_positions == 'scored':
        features = get_scored_activations(
            model, tokenizer, prompts, candidates, labels, q_indices,
            answer_indices, args.layer, device, args.feature_dtype)
        activations = features.pop('activations')
        if category_indices is not None:
            question_categories = dict(zip(q_indices.tolist(), category_indices.tolist()))
            features['category_indices'] = np.asarray(
                [question_categories[int(q)] for q in features['q_indices']])
    else:
        activations = get_layer_activations(model, tokenizer, prompts, args.layer, device)
        if args.feature_dtype == 'float16':
            activations = activations.astype(np.float16)
        features = dict(labels=labels, q_indices=q_indices)

    # 4. Save
    os.makedirs(args.save_dir, exist_ok=True)
    suffix = '_scored' if args.activation_positions == 'scored' else ''
    save_name = f"{args.model_name}_{args.dataset}_{args.split}_l{args.layer}{suffix}.npz"
    save_path = os.path.join(args.save_dir, save_name)

    np.savez(save_path, activations=activations, **features,
             dataset=np.array(args.dataset), split=np.array(args.split),
             data_seed=np.array(args.seed),
             dev_num_samples=np.array(len(np.unique(q_indices))),
             activation_positions=np.array(args.activation_positions),
             prompt_format=np.array(
                 'match-evaluation' if args.activation_positions == 'scored' else
                 'answer-conditioned' if args.dataset == 'boolq' else 'legacy'),
             feature_dtype=np.array(args.feature_dtype))
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
