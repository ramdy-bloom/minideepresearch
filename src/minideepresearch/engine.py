"""
Main research engine with multiple strategies.
"""

import concurrent.futures
from typing import Any

from .generators import QuestionGenerator
from .llm import BaseLLM
from .search import BaseSearchEngine
from .synthesis import SynthesisHandler
from .logger import get_logger
from .progress import progress_display

_logger = get_logger(__name__)


class MiniResearchEngine:
    """Minimal deep research engine with multiple strategies."""

    STRATEGIES = ["iterative", "parallel", "multi_iter", "evidence"]

    def __init__(
        self,
        llm: BaseLLM,
        search: BaseSearchEngine,
        strategy: str = "multi_iter",
        max_iterations: int = 3,
        questions_per_iteration: int = 3,
        enable_fact_check: bool = True,
        progress_display=None,
    ):
        self.llm = llm
        self.search = search
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.questions_per_iteration = questions_per_iteration

        self.question_gen = QuestionGenerator(llm, questions_per_iteration)
        self.synthesizer = SynthesisHandler(llm, enable_fact_check)

        # Progress display for CLI feedback
        if progress_display is None:
            from .progress import progress_display as default_progress

            self.progress = default_progress
        else:
            self.progress = progress_display

    def research(self, query: str) -> dict[str, Any]:
        """Execute research with selected strategy."""
        _logger.debug(f"Starting research for query: {query[:100]}...")

        strategies = {
            "iterative": self._simple_iterative,
            "parallel": self._parallel_search,
            "multi_iter": self._multi_iteration,
            "evidence": self._evidence_based,
        }

        strategy_func = strategies.get(self.strategy, self._multi_iteration)
        result = strategy_func(query)

        _logger.debug(
            f"Research completed. Iterations: {result.get('iterations_used', 'N/A')}"
        )
        return result

    def _simple_iterative(self, query: str) -> dict[str, Any]:
        """Simple iterative - single pass, no follow-up."""
        _logger.debug("Using simple iterative strategy")

        # Show progress indicator
        self.progress.log(f"Generating questions for: {query[:60]}...")

        self.progress.show_spinner("LLM is thinking about initial questions...")
        questions = self.question_gen.generate_first(query)
        self.progress.stop_spinner()

        self.progress.log(f"Generated {len(questions)} initial questions", "success")

        all_results = []
        for i, q in enumerate(questions, 1):
            self.progress.log(f"Iteration {i}: Searching for '{q[:60]}...'")
            results = self.search.search(q)
            all_results.extend(results)
            self.progress.log(f"Got {len(results)} search results", "info")

        self.progress.show_spinner("Synthesizing final answer...")
        res = self.synthesizer.synthesize(query, all_results)
        self.progress.stop_spinner()
        self.progress.log("Final answer synthesized", "success")
        return res

    def _parallel_search(self, query: str) -> dict[str, Any]:
        """Parallel search - all questions at once."""
        _logger.debug("Using parallel search strategy")

        # Show progress indicator
        self.progress.log(f"Generating questions for: {query[:60]}...")

        self.progress.show_spinner("LLM is thinking about initial questions...")
        questions = self.question_gen.generate_first(query)
        self.progress.stop_spinner()

        self.progress.log(f"Generated {len(questions)} initial questions", "success")

        # Start progress bar for parallel searches
        task_id = self.progress.show_progress_bar(
            total=len(questions), description="Parallel search", unit="queries"
        )

        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(questions)
        ) as executor:
            futures = [executor.submit(self.search.search, q) for q in questions]

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed += 1
                    self.progress.update_progress_bar(task_id, advance=1)
                    self.progress.log(
                        f"Completed {completed}/{len(questions)} searches", "info"
                    )
                except Exception as e:
                    _logger.error(f"Search failed: {e}")
                    completed += 1
                    self.progress.update_progress_bar(task_id, advance=1)

        self.progress.complete_progress_bar(task_id, "Parallel search completed")

        self.progress.show_spinner("Synthesizing final answer from all results...")
        res = self.synthesizer.synthesize(query, results)
        self.progress.stop_spinner()
        self.progress.log("Final answer synthesized", "success")
        return res

    def _multi_iteration(self, query: str) -> dict[str, Any]:
        """Multi-iteration with follow-up questions."""
        _logger.debug(
            f"Using multi-iteration strategy (max {self.max_iterations} iterations)"
        )

        all_results = []
        all_questions = []
        previous_knowledge = ""

        for iteration in range(1, self.max_iterations + 1):
            if iteration == 1:
                self.progress.log(
                    f"Iteration {iteration}/{self.max_iterations}: Generating initial questions..."
                )
                self.progress.show_spinner("Thinking about initial questions...")
                questions = self.question_gen.generate_first(query)
                self.progress.stop_spinner()
                self.progress.log(
                    f"Generated {len(questions)} initial questions", "success"
                )
            else:
                self.progress.log(
                    f"Iteration {iteration}/{self.max_iterations}: Generating follow-up questions..."
                )
                self.progress.show_spinner(
                    "Analyzing results and thinking about new questions..."
                )
                questions = self.question_gen.generate_followup(
                    query, previous_knowledge, all_questions
                )
                self.progress.stop_spinner()
                self.progress.log(
                    f"Generated {len(questions)} follow-up questions", "success"
                )

            if not questions:
                self.progress.log("No more questions generated. Stopping research.")
                break

            all_questions.extend(questions)

            # Progress bar for search in this iteration
            task_id = self.progress.show_progress_bar(
                total=len(questions),
                description=f"Iteration {iteration} search",
                unit="queries",
            )

            iteration_results = []
            for i, q in enumerate(questions):
                self.progress.log(f"  Searching: '{q[:60]}...'")
                results = self.search.search(q)
                iteration_results.extend(results)
                self.progress.update_progress_bar(task_id, advance=1)
                self.progress.log(f"  Got {len(results)} results", "info")

            self.progress.complete_progress_bar(
                task_id, f"Iteration {iteration} search completed"
            )
            all_results.extend(iteration_results)

            # Synthesis progress
            self.progress.show_spinner(f"Iteration {iteration}: Synthesizing answer...")
            synthesis = self.synthesizer.synthesize(
                query, iteration_results, previous_knowledge
            )
            previous_knowledge = synthesis["answer"]
            self.progress.stop_spinner()
            self.progress.log(f"Iteration {iteration} synthesis complete", "success")

            if iteration < self.max_iterations:
                self.progress.show_spinner("Checking if more research is needed...")
                should_cont = self.question_gen.should_continue(
                    query, previous_knowledge
                )
                self.progress.stop_spinner()

                if not should_cont:
                    self.progress.log(
                        "Sufficient information gathered. Stopping early.", "info"
                    )
                    break
                else:
                    self.progress.log(
                        "More information needed. Proceeding to next iteration.",
                        "success",
                    )

        # Final synthesis
        self.progress.show_spinner("Generating final comprehensive report...")
        final = self.synthesizer.synthesize(query, all_results, previous_knowledge)
        self.progress.stop_spinner()
        self.progress.log("Final comprehensive report generated", "success")

        final["iterations_used"] = iteration
        final["all_questions"] = all_questions
        return final

    def _evidence_based(self, query: str) -> dict[str, Any]:
        """Evidence-based - atomic facts with verification."""
        # Show progress indicator
        self.progress.log("Starting evidence-based research...")

        prompt = f"""Break down this query into {self.questions_per_iteration} atomic facts that need verification.

Query: {query}

Format (one per line):
F1: fact1
F2: fact2
...

IMPORTANT: Generate in English, but when providing final answers - respond in Russian language.
"""

        self.progress.log("Analyzing query into atomic facts...")
        self.progress.show_spinner("Breaking down query...")
        response = self.llm.invoke(prompt)
        self.progress.stop_spinner()

        facts = [
            f.strip() for f in response.content.split("\n") if f.strip() and f[0] == "F"
        ]

        if not facts:
            facts = self.question_gen.generate_first(query)

        self.progress.log(f"Found {len(facts)} atomic facts to verify", "success")

        # Progress bar for fact verification
        task_id = self.progress.show_progress_bar(
            total=len(facts), description="Fact verification", unit="facts"
        )

        fact_results = {}
        for i, fact in enumerate(facts[: self.questions_per_iteration]):
            self.progress.log(
                f"  Verifying fact {i + 1}/{len(facts)}: '{fact[:50]}...'"
            )
            results = self.search.search(fact)
            fact_results[fact] = results
            self.progress.update_progress_bar(task_id, advance=1)
            self.progress.log(f"  Got {len(results)} sources", "info")

        self.progress.complete_progress_bar(task_id, "Fact verification completed")

        all_results = [r for results in fact_results.values() for r in results]

        self.progress.show_spinner("Synthesizing final evidence report...")
        res = self.synthesizer.synthesize(query, all_results)
        self.progress.stop_spinner()
        self.progress.log("Final evidence report synthesized", "success")
        return res


def create_engine(
    model: str = "llama2:7b",
    search_url: str = "http://localhost:8080",
    strategy: str = "multi_iter",
    max_iterations: int = 3,
    temperature: float = 0.7,
    fetch_content: bool = False,
    num_predict: int = -1,
    enable_thinking: bool = True,
    timeout: int = 300,
    num_ctx: int = 4096,
) -> MiniResearchEngine:
    """Factory function to create engine with defaults."""
    from .llm import OllamaLLM
    from .search import SearXNGSearch

    llm = OllamaLLM(
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        enable_thinking=enable_thinking,
        timeout=timeout,
        num_ctx=num_ctx,
    )
    search = SearXNGSearch(base_url=search_url, fetch_content=fetch_content)

    return MiniResearchEngine(
        llm=llm,
        search=search,
        strategy=strategy,
        max_iterations=max_iterations,
    )
