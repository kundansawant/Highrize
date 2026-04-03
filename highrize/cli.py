"""
highrize CLI — compress prompts and files from the terminal.

Usage:
    highrize compress "your long prompt here"
    highrize compress prompt.txt
    highrize compress image.jpg --provider openai
    highrize compress document.pdf --query "pricing details" --budget 1000
    highrize stats                      # show last session report
    highrize benchmark prompt.txt       # compare all strategies
"""

import sys
import os
import json
import argparse
from pathlib import Path


def _print_result(result, verbose=False):
    bar_len = 30
    filled = int(bar_len * (result.savings_pct / 100))
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  Modality  : {result.modality.value}")
    print(f"  Original  : {result.original_tokens:,} tokens")
    print(f"  Compressed: {result.compressed_tokens:,} tokens")
    print(f"  Saved     : {result.tokens_saved:,} tokens ({result.savings_pct}%)")
    print(f"  [{bar}] {result.savings_pct}%")
    if verbose and isinstance(result.compressed, str):
        print(f"\n--- Compressed output ---\n{result.compressed[:500]}")
        if len(result.compressed) > 500:
            print(f"... [{len(result.compressed) - 500} more chars]")


def cmd_compress(args):
    from highrize import HighRize
    from highrize.models import Modality

    source = args.input
    tp = HighRize(
        model=args.model,
        provider=args.provider,
        doc_token_budget=args.budget,
        doc_query=args.query,
        max_image_size=(args.max_size, args.max_size),
        image_quality=args.quality,
    )

    # Determine if it's a file path or raw text
    if os.path.exists(source):
        ext = source.lower().rsplit(".", 1)[-1]
        modality = None  # auto-detect
    else:
        modality = Modality.TEXT

    result = tp.compress(source, modality=modality)
    _print_result(result, verbose=args.verbose)

    if args.output:
        out = result.compressed
        if isinstance(out, list):
            out = json.dumps(out)
        Path(args.output).write_text(out if isinstance(out, str) else str(out))
        print(f"\n  Output saved to: {args.output}")


def cmd_benchmark(args):
    """Compare all text compression strategies side by side."""
    from highrize.compressors.text import TextCompressor, _estimate_tokens

    source = args.input
    if os.path.exists(source):
        text = Path(source).read_text()
    else:
        text = source

    original_tokens = _estimate_tokens(text)
    print(f"\nhighrize benchmark — {original_tokens} original tokens\n")
    print(f"  {'Strategy':<30} {'Tokens':<12} {'Saved':<10} {'%'}")
    print("  " + "-" * 60)

    strategies = [
        ("No compression", TextCompressor(remove_fillers=False, deduplicate=False)),
        ("Filler removal only", TextCompressor(remove_fillers=True, deduplicate=False)),
        ("Dedup only", TextCompressor(remove_fillers=False, deduplicate=True)),
        ("Filler + dedup", TextCompressor(remove_fillers=True, deduplicate=True)),
        ("Filler + dedup + max 3 examples", TextCompressor(remove_fillers=True, deduplicate=True, max_examples=3)),
    ]

    for name, tc in strategies:
        r = tc.compress(text)
        print(f"  {name:<30} {r.compressed_tokens:<12,} {r.tokens_saved:<10,} {r.savings_pct}%")

    print()


def cmd_count(args):
    """Just count tokens in a string or file."""
    from highrize.tokens import TokenCounter

    source = args.input
    if os.path.exists(source):
        text = Path(source).read_text()
    else:
        text = source

    tc = TokenCounter(model=args.model, provider=args.provider)
    n = tc.count(text)
    print(f"\n  {n:,} tokens  (model: {args.model}, backend: {tc.backend})\n")


def main():
    parser = argparse.ArgumentParser(
        prog="highrize",
        description="Universal AI token compressor",
    )
    sub = parser.add_subparsers(dest="command")

    # compress
    p_compress = sub.add_parser("compress", help="Compress a prompt or file")
    p_compress.add_argument("input", help="Text string or file path")
    p_compress.add_argument("--model", default="gpt-4o")
    p_compress.add_argument("--provider", default="openai")
    p_compress.add_argument("--output", "-o", default=None, help="Save compressed output to file")
    p_compress.add_argument("--query", default=None, help="Relevance query for document compression")
    p_compress.add_argument("--budget", type=int, default=2000, help="Token budget for documents")
    p_compress.add_argument("--max-size", type=int, default=1024, help="Max image dimension")
    p_compress.add_argument("--quality", type=int, default=75, help="Image JPEG quality")
    p_compress.add_argument("--verbose", "-v", action="store_true", help="Show compressed output")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Compare compression strategies")
    p_bench.add_argument("input", help="Text string or file path")

    # count
    p_count = sub.add_parser("count", help="Count tokens in text or file")
    p_count.add_argument("input", help="Text string or file path")
    p_count.add_argument("--model", default="gpt-4o")
    p_count.add_argument("--provider", default="openai")

    args = parser.parse_args()

    if args.command == "compress":
        cmd_compress(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "count":
        cmd_count(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
