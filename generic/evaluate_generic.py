"""
Evaluate MC tasks with Spherical Steering.

Evaluation logic (MC1 style):
  1. For each question, construct full prompts for ALL options.
  2. Calculate log_prob for the option text part only.
  3. Score each option by the sum of its conditional token log-probabilities.
  4. Select option with max score → compare with ground truth.

Supported datasets: COPA, StoryCloze, MMLU, Winogrande, BoolQ

Usage:
    python evaluate_generic.py --model_name llama3.1-8B --dataset copa --layer 14 \
        --prototype_path ./prototypes_generic/xxx.npz --alpha 0.3 --beta -0.2
    python evaluate_generic.py --model_name llama3.1-8B --dataset mmlu_global --layer 14 \
        --prototype_path ./prototypes_generic/xxx.npz --alpha 0.3 --beta -0.2
"""

import argparse
import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from functools import partial
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from baukit import TraceDict

# spherical_steering.py lives in parent directory (ICML2026/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from steering_artifacts import load_steering_artifact, build_intervention
from utils_generic import (HF_NAMES, MMLU_CATEGORIES, set_seed,
                           format_winogrande_eval_prompt, format_mmlu_eval_prompt,
                           format_copa_eval_prompt, get_mmlu_category_questions)
from utils_generic import format_boolq_eval_prompt, format_storycloze_eval_prompt


# ============================================================
# Prompt Formatters
# ============================================================

def format_copa_prompt(premise, question_type, choice1, choice2):
    return format_copa_eval_prompt(premise, question_type, choice1, choice2)


def format_storycloze_prompt(sentences, opt1, opt2):
    return format_storycloze_eval_prompt(sentences, opt1, opt2)


def format_mmlu_prompt(tokenizer, question, choices):
    """Format MMLU prompt using chat template (for Instruct models)."""
    return format_mmlu_eval_prompt(tokenizer, question, choices)


def format_winogrande_prompt(sentence, option1, option2):
    return format_winogrande_eval_prompt(sentence, option1, option2)


def format_boolq_prompt(passage, question):
    return format_boolq_eval_prompt(passage, question)


# ============================================================
# Score Calculation
# ============================================================

def calculate_option_score(model, tokenizer, base_prompt, option_text,
                           hook_fn, layer_name, device, normalize_length=False):
    """
    Calculate log probability score for an option.
    Scores only the option tokens (after the prompt).
    """
    prompt_ids = tokenizer(base_prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    intervention_start_idx = prompt_len - 1

    # Append option text (with leading space for raw prompts, as-is for chat)
    if base_prompt.strip().endswith(":"):
        text_to_append = f" {option_text}"
    else:
        text_to_append = option_text

    option_ids = tokenizer(text_to_append, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    option_len = option_ids.shape[1]
    input_ids = torch.cat([prompt_ids, option_ids], dim=1)

    # Hook setup
    current_hook = partial(hook_fn, start_idx=intervention_start_idx) if hook_fn is not None else None

    with torch.no_grad():
        with TraceDict(model, [layer_name], edit_output=current_hook):
            outputs = model(input_ids)

    logits = outputs.logits
    log_probs_all = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1)
    shift_labels = input_ids[..., 1:]
    token_log_probs = torch.gather(log_probs_all, 2, shift_labels.unsqueeze(2)).squeeze(2)

    option_log_probs = token_log_probs[0, intervention_start_idx:intervention_start_idx + option_len]

    if normalize_length:
        return option_log_probs.mean().item()
    else:
        return option_log_probs.sum().item()


# ============================================================
# Per-dataset Evaluation Functions
# ============================================================

def evaluate_copa(model, tokenizer, hook_fn, layer_name, device, steering_stats,
                  artifact, eval_split='official'):
    """Evaluate on 80 development-validation or 100 untouched test examples."""
    if eval_split == 'dev-validation':
        validation_ids = artifact.get('validation_q_indices')
        if validation_ids is None or not len(validation_ids):
            raise ValueError("artifact has no COPA development-validation split")
        data_seed = int(artifact.get('data_seed') or 42)
        dataset = load_dataset("aps/super_glue", "copa", split="train").shuffle(seed=data_seed)
        dataset = dataset.select([int(value) for value in validation_ids])
        print(f"Loading {len(dataset)} COPA development-validation examples...")
    else:
        print("Loading untouched COPA validation/test split...")
        dataset = load_dataset("aps/super_glue", "copa", split="validation")
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        premise = item["premise"]
        q_type = item["question"]
        choice1 = item["choice1"]
        choice2 = item["choice2"]
        correct_idx = item["label"]  # 0 or 1
        choices = [choice1, choice2]

        base_prompt = format_copa_prompt(premise, q_type, choice1, choice2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, c,
                                         hook_fn, layer_name, device) for c in choices]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def evaluate_storycloze(model, tokenizer, hook_fn, layer_name, device,
                        steering_stats, artifact, eval_split='official'):
    """Evaluate on development-validation or the untouched official eval set."""
    if eval_split == 'dev-validation':
        validation_ids = artifact.get('validation_q_indices')
        if validation_ids is None or not len(validation_ids):
            raise ValueError("artifact has no StoryCloze development-validation split")
        dataset = load_dataset("juletxara/xstory_cloze", "en")["train"]
        expected = int(artifact.get('dev_num_samples') or -1)
        if expected != len(dataset):
            raise ValueError("StoryCloze artifact development size mismatch")
        dataset = dataset.select([int(value) for value in validation_ids])
        print(f"Loading {len(dataset)} StoryCloze development-validation examples...")
    else:
        print("Loading untouched XStoryCloze (en) eval set...")
        dataset = load_dataset("juletxara/xstory_cloze", "en")['eval']
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        story = [item[f'input_sentence_{i}'] for i in range(1, 5)]
        opt1 = item['sentence_quiz1']
        opt2 = item['sentence_quiz2']
        correct_idx = item['answer_right_ending'] - 1  # 1→0, 2→1

        base_prompt = format_storycloze_prompt(story, opt1, opt2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in [opt1, opt2]]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def _evaluate_mmlu_items(model, tokenizer, hook_fn, layer_name, device, items, steering_stats):
    """Shared evaluation loop for MMLU items."""
    correct, total = 0, 0
    category_counts = {}
    pbar = tqdm(items)

    for item in pbar:
        question = item['question']
        choices = item['choices']
        correct_idx = item['answer']

        base_prompt = format_mmlu_prompt(tokenizer, question, choices)
        scores = [calculate_option_score(model, tokenizer, base_prompt, c,
                                         hook_fn, layer_name, device) for c in choices]

        is_correct = int(np.argmax(scores) == correct_idx)
        if is_correct:
            correct += 1
        category = item.get('_category', 'Unknown')
        cat_correct, cat_total = category_counts.get(category, (0, 0))
        category_counts[category] = (cat_correct + is_correct, cat_total + 1)
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    for category, (cat_correct, cat_total) in category_counts.items():
        print(f"  {category}: {cat_correct / cat_total:.4f} ({cat_correct}/{cat_total})")
    return correct / total


def evaluate_mmlu_global(model, tokenizer, hook_fn, layer_name, device,
                         steering_stats, artifact, eval_split='evaluation'):
    """
    Development validation uses 100/category from the 500/category development
    pool. Evaluation uses the next disjoint 200/category.
    """
    print(f"Loading MMLU Global Balanced (split={eval_split})...")

    data_seed = int(artifact.get('data_seed') or 42)
    all_items = []
    if eval_split == 'dev-validation':
        validation_ids = artifact.get('validation_q_indices')
        if validation_ids is None or not len(validation_ids):
            raise ValueError("artifact has no development-validation split")
        selected_ids = {int(value) for value in validation_ids}
        by_category = get_mmlu_category_questions(data_seed, start=0, count=500)
        global_question = 0
        for _, (category_name, dataset) in by_category.items():
            selected = []
            for item in dataset:
                if global_question in selected_ids:
                    row = dict(item); row['_category'] = category_name; selected.append(row)
                global_question += 1
            print(f"  {category_name}: {len(selected)} validation questions")
            all_items.extend(selected)
    elif eval_split in ('evaluation', 'official'):
        by_category = get_mmlu_category_questions(data_seed, start=500, count=200)
        for _, (category_name, dataset) in by_category.items():
            rows = []
            for item in dataset:
                row = dict(item); row['_category'] = category_name; rows.append(row)
            print(f"  {category_name}: {len(rows)} evaluation questions")
            all_items.extend(rows)
    else:
        raise ValueError("MMLU eval split must be dev-validation or evaluation")

    print(f"Total: {len(all_items)} questions")
    return _evaluate_mmlu_items(model, tokenizer, hook_fn, layer_name, device,
                                all_items, steering_stats)


def evaluate_winogrande(model, tokenizer, hook_fn, layer_name, device,
                        steering_stats, artifact, eval_split='official'):
    """Evaluate on development-validation or the full official validation set."""
    if eval_split == 'dev-validation':
        validation_ids = artifact.get('validation_q_indices')
        if validation_ids is None or not len(validation_ids):
            raise ValueError("artifact has no development-validation split")
        data_seed = int(artifact.get('data_seed') or 42)
        n_development = int(artifact.get('dev_num_samples') or -1)
        if n_development <= 0:
            raise ValueError("artifact is missing a valid dev_num_samples value")
        print(f"Loading {len(validation_ids)} WinoGrande development-validation questions...")
        dataset = load_dataset("allenai/winogrande", "winogrande_xl")['train'].shuffle(seed=data_seed)
        dataset = dataset.select(range(min(n_development, len(dataset))))
        dataset = dataset.select([int(i) for i in validation_ids])
    else:
        print("Loading full WinoGrande official validation set...")
        dataset = load_dataset("allenai/winogrande", "winogrande_xl")['validation']
    print(f"Evaluating on {len(dataset)} samples...")

    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        sentence = item['sentence']
        opt1 = item['option1']
        opt2 = item['option2']
        correct_idx = int(item['answer']) - 1  # '1'→0, '2'→1

        base_prompt = format_winogrande_prompt(sentence, opt1, opt2)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in [opt1, opt2]]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


def evaluate_boolq(model, tokenizer, hook_fn, layer_name, device, steering_stats,
                   artifact, eval_split='official'):
    """Evaluate on 200 development-validation or 3,270 official examples."""
    if eval_split == 'dev-validation':
        validation_ids = artifact.get('validation_q_indices')
        if validation_ids is None or not len(validation_ids):
            raise ValueError("artifact has no BoolQ development-validation split")
        data_seed = int(artifact.get('data_seed') or 42)
        n_development = int(artifact.get('dev_num_samples') or -1)
        if n_development != 1000:
            raise ValueError("BoolQ artifact must contain exactly 1,000 development questions")
        dataset = load_dataset("aps/super_glue", "boolq", split="train").shuffle(seed=data_seed)
        dataset = dataset.select(range(n_development))
        dataset = dataset.select([int(value) for value in validation_ids])
        print(f"Loading {len(dataset)} BoolQ development-validation examples...")
    else:
        print("Loading full official BoolQ validation set...")
        dataset = load_dataset("aps/super_glue", "boolq", split="validation")
    print(f"Evaluating on {len(dataset)} samples...")

    options = ["no", "yes"]
    correct, total = 0, 0
    pbar = tqdm(dataset)

    for item in pbar:
        passage = item["passage"]
        question = item["question"]
        correct_idx = int(item["label"])  # 0→no, 1→yes

        base_prompt = format_boolq_prompt(passage, question)
        scores = [calculate_option_score(model, tokenizer, base_prompt, opt,
                                         hook_fn, layer_name, device) for opt in options]

        if np.argmax(scores) == correct_idx:
            correct += 1
        total += 1
        pbar.set_description(f"Acc: {correct/total:.4f}")

    return correct / total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate MC tasks with Spherical Steering")
    parser.add_argument('--model_name', type=str, default='llama3.1-8B-Instruct',
                        help=f"Model name or path. Shortcuts: {list(HF_NAMES.keys())}")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['copa', 'storycloze', 'mmlu_global', 'winogrande', 'boolq'])
    parser.add_argument('--layer', type=int, default=14)
    parser.add_argument('--prototype_path', type=str, required=True)
    parser.add_argument('--kappa', type=float, default=20.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=-0.2)
    parser.add_argument('--disable_steering', action='store_true')
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Local model directory (overrides model_name)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steering-geometry', choices=['auto', 'sphere', 'ellipsoid'], default='auto')
    parser.add_argument('--eval-split', choices=['official', 'dev-validation', 'evaluation'],
                        default='official', help='development selection or frozen evaluation split')

    args = parser.parse_args()
    if args.eval_split == 'dev-validation' and args.dataset not in (
            'winogrande', 'mmlu_global', 'copa', 'boolq', 'storycloze'):
        parser.error('--eval-split dev-validation is unsupported for this dataset')
    if args.eval_split == 'evaluation' and args.dataset != 'mmlu_global':
        parser.error('--eval-split evaluation is specific to MMLU')
    set_seed(args.seed)

    # Load model
    model_path = args.model_dir if args.model_dir else HF_NAMES.get(args.model_name, args.model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prototypes
    print(f"Loading prototypes: {args.prototype_path}")
    artifact = load_steering_artifact(args.prototype_path, device)

    # Setup steering hook
    layer_name = f"model.layers.{args.layer}"
    steering_stats = {'total': 0, 'steered': 0}

    if args.disable_steering:
        print(">> Mode: BASELINE (No Steering)")
        hook_fn = None
    else:
        print(f">> Mode: STEERING (kappa={args.kappa}, alpha={args.alpha}, beta={args.beta})")
        hook_fn = build_intervention(
            artifact, args.kappa, args.alpha, args.beta, stats=steering_stats,
            steering_geometry=args.steering_geometry)

    # Dispatch evaluation
    if args.dataset == 'copa':
        accuracy = evaluate_copa(model, tokenizer, hook_fn, layer_name, device,
                                 steering_stats, artifact, args.eval_split)
    elif args.dataset == 'storycloze':
        accuracy = evaluate_storycloze(model, tokenizer, hook_fn, layer_name, device,
                                       steering_stats, artifact, args.eval_split)
    elif args.dataset == 'mmlu_global':
        accuracy = evaluate_mmlu_global(model, tokenizer, hook_fn, layer_name, device,
                                        steering_stats, artifact, args.eval_split)
    elif args.dataset == 'winogrande':
        accuracy = evaluate_winogrande(model, tokenizer, hook_fn, layer_name, device,
                                       steering_stats, artifact, args.eval_split)
    elif args.dataset == 'boolq':
        accuracy = evaluate_boolq(model, tokenizer, hook_fn, layer_name, device,
                                  steering_stats, artifact, args.eval_split)

    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Dataset:    {args.dataset}")
    if args.dataset in ('winogrande', 'mmlu_global', 'copa', 'boolq', 'storycloze'):
        print(f"Eval split: {args.eval_split}")
    print(f"Model:      {args.model_name}")
    print(f"Layer:      {args.layer}")
    print(f"Steering:   {'OFF' if args.disable_steering else 'ON'}")
    if not args.disable_steering:
        print(f"Parameters: kappa={args.kappa}, alpha={args.alpha}, beta={args.beta}")
        if steering_stats['total'] > 0:
            steer_pct = steering_stats['steered'] / steering_stats['total'] * 100
            print(f"Stats:      Steered={steering_stats['steered']}/{steering_stats['total']} ({steer_pct:.1f}%)")
    print("-" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
