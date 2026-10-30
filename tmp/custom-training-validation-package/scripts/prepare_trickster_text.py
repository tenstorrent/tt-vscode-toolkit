#!/usr/bin/env python3
"""
Prepare trickster dataset for character-level training.
Converts JSONL Q&A format to plain text suitable for NanoGPT-style training.
"""

import json
import sys

def prepare_trickster_text(input_jsonl, output_txt):
    """Convert trickster JSONL to plain text with special formatting."""

    examples = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            prompt = item['prompt']
            response = item['response']

            # Format as Q&A with clear delimiters for the model to learn
            example = f"Q: {prompt}\nA: {response}\n\n"
            examples.append(example)

    # Combine all examples
    full_text = ''.join(examples)

    # Write to output file
    with open(output_txt, 'w') as f:
        f.write(full_text)

    # Statistics
    num_examples = len(examples)
    num_chars = len(full_text)
    vocab = sorted(set(full_text))
    vocab_size = len(vocab)

    print(f"✅ Prepared trickster dataset:")
    print(f"   Examples: {num_examples}")
    print(f"   Total characters: {num_chars:,}")
    print(f"   Vocabulary size: {vocab_size} unique characters")
    print(f"   Output: {output_txt}")

    return num_chars, vocab_size

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'trickster_dataset_starter.jsonl'
    output_file = 'trickster_text.txt'

    prepare_trickster_text(input_file, output_file)
