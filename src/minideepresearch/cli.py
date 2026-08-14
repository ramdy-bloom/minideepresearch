#!/usr/bin/env python3
"""CLI for MiniDeepResearch."""

import argparse
import logging
import sys

from minideepresearch import create_engine, setup_logger
from minideepresearch.formatter import format_json, format_markdown, format_text
from minideepresearch.progress import progress_display

logger = setup_logger("minideepresearch-cli")


def main():
    parser = argparse.ArgumentParser(description="MiniDeepResearch CLI")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--model", "-m", default="llama2:7b", help="Ollama model name")
    parser.add_argument(
        "--search-url",
        "-s",
        default="http://localhost:8080",
        help="SearXNG URL",
    )
    parser.add_argument(
        "--strategy",
        "-t",
        choices=["iterative", "parallel", "multi_iter", "evidence"],
        default="multi_iter",
        help="Research strategy",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=3,
        help="Max iterations (for multi_iter)",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="LLM temperature",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=-1,
        help="Max tokens to generate (-1 = unlimited). Lower values reduce 'thinking' time.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="LLM request timeout in seconds",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=32768,
        help="LLM context window size",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable model thinking/reasoning (for Qwen3.5 models)",
    )
    parser.add_argument(
        "--no-fact-check",
        action="store_true",
        help="Disable fact checking",
    )
    parser.add_argument(
        "--fetch",
        "-f",
        action="store_true",
        help="Fetch full content from result URLs (slower but more detailed)",
    )
    parser.add_argument(
        "--fetch-backend",
        choices=["auto", "requests", "curl"],
        default="auto",
        help="Fetch backend: auto = requests with curl fallback, requests only, or curl only",
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--markdown", "-md", action="store_true", help="Output as Markdown"
    )
    parser.add_argument("--output", "-o", help="Save output to file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output (DEBUG level)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (errors only)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("minideepresearch").setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger("minideepresearch").setLevel(logging.ERROR)

    logger.debug(f"Starting research with model={args.model}, strategy={args.strategy}")

    progress_display.log("=" * 50, "info")
    progress_display.log(f"Model: [bold cyan]{args.model}[/bold cyan]", "info")
    progress_display.log(
        f"Strategy: [bold magenta]{args.strategy}[/bold magenta]", "info"
    )
    progress_display.log(f"Query: [italic]{args.query}[/italic]", "info")
    progress_display.log("=" * 50, "info")

    engine = create_engine(
        model=args.model,
        search_url=args.search_url,
        strategy=args.strategy,
        max_iterations=args.iterations,
        temperature=args.temperature,
        fetch_content=args.fetch,
        fetch_backend=args.fetch_backend,
        num_predict=args.num_predict,
        enable_thinking=not args.no_thinking,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
        enable_fact_check=not args.no_fact_check,
    )

    try:
        logger.debug("Starting research")
        result = engine.research(args.query)
        logger.debug("Research completed")

        if args.json:
            output = format_json(result)
        elif args.markdown:
            output = format_markdown(result)
        elif args.output:
            output = format_text(result)
        else:
            output = ""
            progress_display.log("\n" + "=" * 60, "info")
            progress_display.log("ANSWER:", "success")
            progress_display.log("=" * 60, "info")
            progress_display.print_result("Answer", result["answer"])

            progress_display.log("\n" + "=" * 60, "info")
            progress_display.log("SOURCES:", "success")
            progress_display.log("=" * 60, "info")

            for i, src in enumerate(result["sources"], 1):
                progress_display.log(f"{i}. {src.get('title', 'No title')}", "info")
                progress_display.log(f"   {src.get('url', '')}", "info")

            if result.get("iterations_used"):
                progress_display.log(
                    f"\nIterations used: {result['iterations_used']}", "info"
                )
            if result.get("all_questions"):
                progress_display.log(
                    f"Questions generated: {len(result['all_questions'])}", "info"
                )
            if result.get("fact_check"):
                progress_display.log(f"Fact check: {result['fact_check']}", "info")

        if args.output and output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            progress_display.log(f"\nSaved to: {args.output}", "success")
        elif output and not args.output:
            print(output)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
