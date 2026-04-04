"""
Synthesis handler for generating answers with citations.
"""

from typing import Any

from .llm import BaseLLM
from .logger import get_logger
from .errors import retry, validate_llm_response

_logger = get_logger(__name__)


class SynthesisHandler:
    """Handles synthesis of search results into final answer."""

    def __init__(self, llm: BaseLLM, enable_fact_check: bool = True):
        self.llm = llm
        self.enable_fact_check = enable_fact_check

    @retry(max_attempts=3, delay=2.0, backoff=1.5)
    def synthesize(
        self,
        query: str,
        results: list[dict],
        previous_knowledge: str = "",
        skip_fact_check: bool = False,
    ) -> dict[str, Any]:
        """Synthesize an answer from search results."""
        _logger.debug(f"Synthesizing answer for query: {query[:80]}...")
        _logger.debug(f"Using {len(results)} sources")

        sources = self._format_sources(results)

        fact_check_notes = ""
        if self.enable_fact_check and not skip_fact_check and previous_knowledge:
            _logger.debug("Running fact check")
            fact_check_notes = self._fact_check(previous_knowledge, sources)

        prompt = self._build_synthesis_prompt(
            query=query,
            sources=sources,
            previous_knowledge=previous_knowledge,
            fact_check_notes=fact_check_notes,
        )

        response = self.llm.invoke(prompt)

        # Validate LLM response
        validate_llm_response(response.content, "synthesizer")

        _logger.debug("Synthesis complete")

        return {
            "answer": response.content,
            "sources": results,
            "fact_check": fact_check_notes if fact_check_notes else None,
        }

    def _build_synthesis_prompt(
        self,
        query: str,
        sources: str,
        previous_knowledge: str = "",
        fact_check_notes: str = "",
    ) -> str:
        """Build the synthesis prompt."""
        prompt = f"""You are an expert technical researcher. Your task is to provide a DEEP and COMPREHENSIVE technical report on the given query based on the provided sources.

### QUERY:
{query}

"""

        if previous_knowledge:
            prompt += f"""### PREVIOUS FINDINGS (Incorporate and Expand):
{previous_knowledge}

"""

        prompt += f"""### NEW SOURCES:
{sources}

"""

        if fact_check_notes:
            prompt += f"""### DISCREPANCIES TO RESOLVE:
{fact_check_notes}

"""

        prompt += """### INSTRUCTIONS FOR THE REPORT:
1. **Depth & Breadth**: Do not just summarize. Analyze the underlying technologies, architectures, and methodologies. Provide specific examples where possible.
2. **Structure**: Use the following structure for your report:
   - **Executive Summary**: A high-level overview of the findings.
   - **Technical Deep Dive**: Detailed analysis of methods, algorithms, and architectures.
   - **Practical Implementation**: How to apply this in practice (libraries, code snippets, configurations).
   - **Comparison/Analysis**: Pros/cons, trade-offs, or comparison of different approaches.
   - **Conclusion & Recommendations**: Final thoughts and suggested path forward.
3. **Citations**: Every technical fact or claim MUST be cited using [1], [2], etc.
4. **Language**: Write the entire report in Russian language.
5. **Formatting**: Use clean Markdown with headers, lists, and bold text for readability.

Your goal is to provide a final, "publish-ready" research document.

REPORT IN RUSSIAN:"""

        return prompt

    def _fact_check(self, previous: str, new_sources: str) -> str:
        """Check for contradictions between sources."""
        _logger.debug("Running fact check")

        prompt = f"""Quickly check if there are any contradictions between the previous
knowledge and the new sources.

Previous Knowledge:
{previous[:1500]}

New Sources:
{new_sources[:1500]}

Respond with either:
- "Consistent" if no contradictions
- Brief note of any contradictions found

IMPORTANT: Write your response in Russian language."""

        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            return result if result and result != "Consistent" else ""
        except Exception:
            return ""

    def _format_sources(self, results: list[dict]) -> str:
        """Format search results for the prompt."""
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("snippet", "")

            if snippet:
                formatted.append(f"[{i}] {title}\n{snippet}\nURL: {url}")
            else:
                formatted.append(f"[{i}] {title}\nURL: {url}")

        return "\n\n".join(formatted)
