"""
Inference script for GPT-Neo 28M model trained on TinyStories.
"""
import argparse
import json
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM, GPTNeoConfig

from utils import VocabRemapper, load_checkpoint


def generate_text(
    model,
    tokenizer,
    vocab_remapper,
    prompt,
    max_length=200,
    temperature=1.0,
    top_k=50,
    device='cuda'
):
    """
    Generate text from the model.

    Args:
        model: Trained GPT-Neo model
        tokenizer: Original GPT-Neo tokenizer
        vocab_remapper: VocabRemapper instance
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to use

    Returns:
        Generated text string
    """
    model.eval()

    # Encode prompt with original tokenizer
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Remap to reduced vocabulary
    input_ids = vocab_remapper.remap_tokens(input_ids).to(device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=vocab_remapper.vocab_data['special_tokens']['pad_token_id'],
            eos_token_id=vocab_remapper.vocab_data['special_tokens']['eos_token_id'],
        )

    # Reverse remap to original vocabulary for decoding
    output_ids = vocab_remapper.reverse_remap_tokens(output_ids.cpu())

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-Neo 28M")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config-28M-gptneo.json',
                        help='Path to model config')
    parser.add_argument('--vocab_mapping', type=str, default='vocab_mapping.json',
                        help='Path to vocabulary mapping')
    parser.add_argument('--prompt', type=str,
                        default="One day, a little girl named Lily found a needle in her room.",
                        help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=200,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load vocabulary mapping
    print(f"Loading vocabulary mapping from {args.vocab_mapping}")
    vocab_remapper = VocabRemapper(args.vocab_mapping)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

    # Load model config
    print(f"Loading model config from {args.config}")
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    config = GPTNeoConfig(**config_dict)

    # Initialize model
    print("Initializing model...")
    model = GPTNeoForCausalLM(config)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(args.checkpoint, model)

    model.to(device)
    model.eval()

    # Generate
    print("\n" + "="*80)
    print(f"Prompt: {args.prompt}")
    print("="*80)

    for i in range(args.num_samples):
        print(f"\nSample {i+1}:")
        print("-"*80)

        generated = generate_text(
            model,
            tokenizer,
            vocab_remapper,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        print(generated)
        print()

    print("="*80)


if __name__ == "__main__":
    main()
