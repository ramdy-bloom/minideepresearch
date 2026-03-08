#!/usr/bin/env python3
"""CLI for MiniDeepResearch."""

import argparse
import json
import logging
import sys

from minideepresearch import create_engine, setup_logger
from minideepresearch.progress import progress_display

# Setup logging
logger = setup_logger("minideepresearch-cli")


def main():
    parser = argparse.ArgumentParser(description="MiniDeepResearch CLI")
    parser.add_argument("query", help="Research query")
    parser.add_argument(
        "--model", "-m", default="llama2:7b", help="Ollama model name"
    )
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
        "--json", "-j", action="store_true", help="Output as JSON"
    )
    parser.add_argument(
        "--markdown", "-md", action="store_true", help="Output as Markdown"
    )
    parser.add_argument("--output", "-o", help="Save output to file")

    args = parser.parse_args()

    logger.debug(f"Starting research with model={args.model}, strategy={args.strategy}")
    
    # Use progress display for better UX
    progress_display.log("=" * 50, "info")
    progress_display.log(f"Model: [bold cyan]{args.model}[/bold cyan]", "info")
    progress_display.log(f"Strategy: [bold magenta]{args.strategy}[/bold magenta]", "info")
    progress_display.log(f"Query: [italic]{args.query}[/italic]", "info")
    progress_display.log("=" * 50, "info")

    engine = create_engine(
        model=args.model,
        search_url=args.search_url,
        strategy=args.strategy,
        max_iterations=args.iterations,
        temperature=args.temperature,
        fetch_content=args.fetch,
        num_predict=args.num_predict,
        enable_thinking=not args.no_thinking,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
    )
    engine.synthesizer.enable_fact_check = not args.no_fact_check

    try:
        logger.debug("Starting research")
        result = engine.research(args.query)
        logger.debug("Research completed")

        # Format output
        if args.json:
            output = json.dumps(result, indent=2, ensure_ascii=False)
        elif args.markdown:
            md = ["# Research Result\n"]
            md.append(result["answer"])
            md.append("\n---\n\n## Sources\n")
            for i, src in enumerate(result["sources"], 1):
                title = src.get("title", "No title")
                url = src.get("url", "")
                if url:
                    md.append(f"{i}. [{title}]({url})")
                else:
                    md.append(f"{i}. {title}")

            if result.get("iterations_used"):
                md.append(
                    f"\n---\n\n**Iterations:** {result['iterations_used']}"
                )
            if result.get("all_questions"):
                md.append(f"\n**Questions:** {len(result['all_questions'])}")
            if result.get("fact_check"):
                md.append(f"\n**Fact Check:** {result['fact_check']}")

            output = "\n".join(md)
        else:
            # Use progress display for output formatting
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
                progress_display.log(f"\nIterations used: {result['iterations_used']}", "info")
            if result.get("all_questions"):
                progress_display.log(
                    f"Questions generated: {len(result['all_questions'])}", "info"
                )
            if result.get("fact_check"):
                progress_display.log(f"Fact check: {result['fact_check']}", "info")

            # For file output, we need to collect everything into a string
            if args.output:
                lines = [
                    "=" * 60,
                    "ANSWER:",
                    "=" * 60,
                    result["answer"],
                    "\nSOURCES:",
                    "=" * 60
                ]
                for i, src in enumerate(result["sources"], 1):
                    lines.append(f"{i}. {src.get('title', 'No title')} ({src.get('url', '')})")
                
                output = "\n".join(lines)
            else:
                output = ""

        # Output to file or terminal (only if not already printed by progress_display)
        if args.output and output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            progress_display.log(f"\nSaved to: {args.output}", "success")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
