#!/usr/bin/env python3
"""
Dataset Validator for Zen Master Training

Validates JSONL format datasets for fine-tuning.
Checks for:
- Valid JSON structure
- Required fields (prompt, response)
- Reasonable lengths
- Duplicates
- Character encoding issues

Usage:
    python validate_dataset.py zen_dataset_starter.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple


def load_dataset(file_path: Path) -> Tuple[List[Dict], List[str]]:
    """
    Load JSONL dataset and collect any errors.

    Returns:
        Tuple of (valid_examples, error_messages)
    """
    examples = []
    errors = []

    if not file_path.exists():
        errors.append(f"❌ File not found: {file_path}")
        return examples, errors

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")

    return examples, errors


def validate_structure(examples: List[Dict]) -> List[str]:
    """
    Validate that each example has required fields.
    """
    errors = []

    for i, example in enumerate(examples, 1):
        # Check required fields
        if 'prompt' not in example:
            errors.append(f"Example {i}: Missing 'prompt' field")
        if 'response' not in example:
            errors.append(f"Example {i}: Missing 'response' field")

        # Check field types
        if 'prompt' in example and not isinstance(example['prompt'], str):
            errors.append(f"Example {i}: 'prompt' must be a string")
        if 'response' in example and not isinstance(example['response'], str):
            errors.append(f"Example {i}: 'response' must be a string")

        # Check for empty strings
        if 'prompt' in example and not example['prompt'].strip():
            errors.append(f"Example {i}: 'prompt' is empty")
        if 'response' in example and not example['response'].strip():
            errors.append(f"Example {i}: 'response' is empty")

    return errors


def check_lengths(examples: List[Dict]) -> Dict:
    """
    Analyze prompt and response lengths.
    """
    prompt_lengths = [len(ex['prompt']) for ex in examples if 'prompt' in ex]
    response_lengths = [len(ex['response']) for ex in examples if 'response' in ex]

    stats = {
        'prompt_min': min(prompt_lengths) if prompt_lengths else 0,
        'prompt_max': max(prompt_lengths) if prompt_lengths else 0,
        'prompt_avg': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        'response_min': min(response_lengths) if response_lengths else 0,
        'response_max': max(response_lengths) if response_lengths else 0,
        'response_avg': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
    }

    # Check for unusually long entries
    warnings = []
    if stats['prompt_max'] > 500:
        warnings.append(f"⚠️  Some prompts are very long (max: {stats['prompt_max']} chars)")
    if stats['response_max'] > 500:
        warnings.append(f"⚠️  Some responses are very long (max: {stats['response_max']} chars)")

    return stats, warnings


def check_duplicates(examples: List[Dict]) -> List[str]:
    """
    Check for duplicate prompts or responses.
    """
    warnings = []

    prompts = [ex['prompt'] for ex in examples if 'prompt' in ex]
    responses = [ex['response'] for ex in examples if 'response' in ex]

    prompt_counts = Counter(prompts)
    response_counts = Counter(responses)

    duplicate_prompts = {p: c for p, c in prompt_counts.items() if c > 1}
    duplicate_responses = {r: c for r, c in response_counts.items() if c > 1}

    if duplicate_prompts:
        warnings.append(f"⚠️  Found {len(duplicate_prompts)} duplicate prompts")
        for prompt, count in list(duplicate_prompts.items())[:3]:
            warnings.append(f"   '{prompt[:50]}...' appears {count} times")

    if duplicate_responses:
        warnings.append(f"⚠️  Found {len(duplicate_responses)} duplicate responses")
        for response, count in list(duplicate_responses.items())[:3]:
            warnings.append(f"   '{response[:50]}...' appears {count} times")

    return warnings


def check_diversity(examples: List[Dict]) -> List[str]:
    """
    Check for diversity in prompts and responses.
    """
    insights = []

    # Check for question marks in prompts (should be high for Q&A format)
    prompts_with_questions = sum(1 for ex in examples if '?' in ex.get('prompt', ''))
    question_ratio = prompts_with_questions / len(examples) if examples else 0

    insights.append(f"📊 {prompts_with_questions}/{len(examples)} prompts are questions ({question_ratio*100:.1f}%)")

    # Check response variety by looking at unique first words
    first_words = []
    for ex in examples:
        response = ex.get('response', '')
        if response:
            first_word = response.split()[0].lower() if response.split() else ''
            first_words.append(first_word)

    unique_first_words = len(set(first_words))
    insights.append(f"📊 {unique_first_words} unique first words in responses")

    if unique_first_words < len(examples) * 0.3:
        insights.append(f"⚠️  Low response diversity - many responses start similarly")

    return insights


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <dataset.jsonl>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    print("=" * 60)
    print("🔍 Zen Master Dataset Validator")
    print("=" * 60)
    print(f"\nValidating: {file_path}")
    print()

    # Load dataset
    examples, load_errors = load_dataset(file_path)

    if load_errors:
        print("❌ LOADING ERRORS:")
        for error in load_errors:
            print(f"   {error}")
        if not examples:
            print("\n❌ No valid examples found. Cannot continue validation.")
            sys.exit(1)
        print()

    print(f"✅ Loaded {len(examples)} examples")
    print()

    # Validate structure
    structure_errors = validate_structure(examples)
    if structure_errors:
        print("❌ STRUCTURE ERRORS:")
        for error in structure_errors[:10]:  # Show first 10
            print(f"   {error}")
        if len(structure_errors) > 10:
            print(f"   ... and {len(structure_errors) - 10} more errors")
        print()
    else:
        print("✅ All examples have valid structure")
        print()

    # Check lengths
    stats, length_warnings = check_lengths(examples)
    print("📏 LENGTH STATISTICS:")
    print(f"   Prompts:   min={stats['prompt_min']:3d}  max={stats['prompt_max']:3d}  avg={stats['prompt_avg']:6.1f}")
    print(f"   Responses: min={stats['response_min']:3d}  max={stats['response_max']:3d}  avg={stats['response_avg']:6.1f}")
    print()

    if length_warnings:
        for warning in length_warnings:
            print(f"   {warning}")
        print()

    # Check duplicates
    duplicate_warnings = check_duplicates(examples)
    if duplicate_warnings:
        print("🔁 DUPLICATES:")
        for warning in duplicate_warnings:
            print(f"   {warning}")
        print()

    # Check diversity
    diversity_insights = check_diversity(examples)
    if diversity_insights:
        print("🌈 DIVERSITY INSIGHTS:")
        for insight in diversity_insights:
            print(f"   {insight}")
        print()

    # Sample examples
    print("📝 SAMPLE EXAMPLES:")
    for i in range(min(3, len(examples))):
        print(f"\n   Example {i+1}:")
        print(f"   Q: {examples[i].get('prompt', 'N/A')}")
        print(f"   A: {examples[i].get('response', 'N/A')}")
    print()

    # Final verdict
    print("=" * 60)
    total_errors = len(load_errors) + len(structure_errors)
    if total_errors == 0:
        print("✅ VALIDATION PASSED!")
        print(f"   Dataset is ready for training with {len(examples)} examples")
    else:
        print(f"❌ VALIDATION FAILED with {total_errors} errors")
        print("   Fix the errors above before using this dataset")
    print("=" * 60)

    sys.exit(0 if total_errors == 0 else 1)


if __name__ == '__main__':
    main()
