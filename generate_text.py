"""Section 7: load a checkpoint and generate text with the decoding function."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.decoding import decoding
from cs336_basics.transformer_lm import TransformerLM

SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_PROMPT = "Once upon a time"


def build_model(args) -> TransformerLM:
    return TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--vocab-file", type=str, default="data/tinystories_vocab.json")
    parser.add_argument("--merges-file", type=str, default="data/tinystories_merges.txt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", type=str, default="swiglu", choices=["swiglu", "silu"])

    parser.add_argument("--output", type=str, default=None, help="Optional path to save generated text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file, special_tokens=[SPECIAL_TOKEN])
    end_token_id = tokenizer.token_to_id[SPECIAL_TOKEN.encode("utf-8")]

    model = build_model(args).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    output_ids = decoding(
        model,
        prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token_id=end_token_id,
    )
    generated = tokenizer.decode(output_ids[0].tolist())

    print("=== prompt ===")
    print(args.prompt)
    print("\n=== generated ===")
    print(generated)

    if args.output:
        Path(args.output).write_text(generated, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
