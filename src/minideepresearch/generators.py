"""
Question generators for research.
"""

from datetime import datetime
from typing import Any

from .llm import BaseLLM
from .logger import get_logger
from .errors import retry, validate_questions

_logger = get_logger(__name__)


class QuestionGenerator:
    """Generates search questions from queries and previous results."""

    def __init__(self, llm: BaseLLM, num_questions: int = 3):
        self.llm = llm
        self.num_questions = num_questions

    @retry(max_attempts=3, delay=2.0, backoff=1.5)
    def generate_first(self, query: str) -> list[str]:
        """Generate initial search queries for a query."""
        _logger.debug(f"Generating initial search queries for: {query[:80]}...")
        
        prompt = f"""Generate {self.num_questions} high-quality search queries (keywords/phrases) for: {query}

Consider: What specific keywords or technical terms are needed to find information to answer this query?
Avoid natural language questions like "What is...". Use direct search terms.
Today: {datetime.now().strftime("%Y-%m-%d")}

Format (one query per line, start with "Q:"):
Q: search query 1
Q: search query 2
Q: search query 3

IMPORTANT: Generate queries in English to search for information, but when providing final answers - respond in Russian language."""

        response = self.llm.invoke(prompt)
        questions = self._parse_questions(response.content)
        
        # Validate generated questions
        validate_questions(questions, self.num_questions)
        
        _logger.debug(f"Generated {len(questions)} search queries")
        return questions

    @retry(max_attempts=3, delay=2.0, backoff=1.5)
    def generate_followup(
        self,
        query: str,
        previous_results: str,
        past_questions: list[str],
    ) -> list[str]:
        """Generate follow-up search queries based on previous results."""
        _logger.debug(f"Generating {self.num_questions} follow-up search queries")
        
        past_q_str = "\n".join(f"- {q}" for q in past_questions)

        prompt = f"""Based on the original query and previous search results,
generate {self.num_questions} follow-up search queries (keywords/phrases) that remain unanswered.

Original Query: {query}

Previous Queries (avoid duplicates):
{past_q_str}

Previous Results Summary:
{previous_results[:3000]}

Avoid natural language questions. Use technical terms and direct search phrases.

Format (one query per line):
Q: search query 1
Q: search query 2
Q: search query 3

IMPORTANT: Generate queries in English to search for information, but when providing final answers - respond in Russian language."""

        response = self.llm.invoke(prompt)
        questions = self._parse_questions(response.content)
        
        # Validate generated questions
        validate_questions(questions, self.num_questions)
        
        _logger.debug(f"Generated {len(questions)} follow-up search queries")
        return questions

    @retry(max_attempts=3, delay=2.0, backoff=1.5)
    def should_continue(self, query: str, current_knowledge: str) -> bool:
        """Check if more research is needed."""
        _logger.debug("Checking if research should continue")
        
        prompt = f"""Based on the current research, determine if more information is needed.

Original Query: {query}

Current Knowledge Summary:
{current_knowledge[:2000]}

Respond with ONLY one word: YES or NO

YES = need more information (continue research)
NO = have enough information (stop research)

IMPORTANT: When providing final answers - respond in Russian language."""

        response = self.llm.invoke(prompt, min_length=1)
        answer = response.content.strip().upper()
        
        _logger.debug(f"Research continuation decision: {answer}")
        
        # Поддержка и английских (YES/NO) и русских (ДА/НЕТ) ответов
        return answer in ["YES", "ДА"]

    def _parse_questions(self, text: str) -> list[str]:
        """Parse questions from LLM response."""
        questions = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Q:") or line.startswith("Q "):
                q = line.replace("Q:", "").replace("Q ", "").strip()
                if q:
                    questions.append(q)
        return questions[: self.num_questions]
