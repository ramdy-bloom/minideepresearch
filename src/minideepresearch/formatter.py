"""Output formatting for MiniDeepResearch results."""

import json
from typing import Any


def format_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)


def format_markdown(result: dict[str, Any]) -> str:
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
        md.append(f"\n---\n\n**Iterations:** {result['iterations_used']}")
    if result.get("all_questions"):
        md.append(f"\n**Questions:** {len(result['all_questions'])}")
    if result.get("fact_check"):
        md.append(f"\n**Fact Check:** {result['fact_check']}")

    return "\n".join(md)


def format_text(result: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "ANSWER:",
        "=" * 60,
        result["answer"],
        "\n" + "=" * 60,
        "SOURCES:",
        "=" * 60,
    ]
    for i, src in enumerate(result["sources"], 1):
        lines.append(f"{i}. {src.get('title', 'No title')} ({src.get('url', '')})")

    if result.get("iterations_used"):
        lines.append(f"\nIterations used: {result['iterations_used']}")
    if result.get("all_questions"):
        lines.append(f"Questions generated: {len(result['all_questions'])}")
    if result.get("fact_check"):
        lines.append(f"Fact check: {result['fact_check']}")

    return "\n".join(lines)
