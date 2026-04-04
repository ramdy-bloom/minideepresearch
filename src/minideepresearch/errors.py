"""
Error handling utilities for MiniDeepResearch.
"""

import functools
import time
from typing import Callable, Any, TypeVar, Optional, NoReturn

from .logger import get_logger

_logger = get_logger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying failed operations.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay between retries
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception = RuntimeError("No attempts made")

            for attempt in range(1, max_attempts + 1):
                try:
                    _logger.debug(f"Attempt {attempt}/{max_attempts}: {func.__name__}")
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait_time = delay * (backoff ** (attempt - 1))
                        _logger.warning(
                            f"{func.__name__} failed (attempt {attempt}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        _logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator


def validate_llm_response(
    response_content: str, model_name: str, min_length: int = 10
) -> None:
    """
    Validate LLM response content.

    Args:
        response_content: Response text from LLM
        model_name: Name of the model that generated response
        min_length: Minimum allowed length for the response

    Raises:
        ValueError: If response is empty or invalid
    """
    if not response_content or not response_content.strip():
        _logger.error(f"LLM ({model_name}) returned empty response")
        raise ValueError("LLM returned empty response")

    if len(response_content) < min_length:
        _logger.warning(
            f"LLM ({model_name}) returned very short response "
            f"({len(response_content)} chars)"
        )
        raise ValueError(
            f"LLM returned very short response: {len(response_content)} chars"
        )

    # Check for common issues
    if (
        "I don't know" in response_content.lower()
        or "cannot answer" in response_content.lower()
    ):
        _logger.warning(f"LLM ({model_name}) indicated inability to answer")


def validate_questions(questions: list[str], expected_count: int) -> None:
    """
    Validate generated questions.

    Args:
        questions: List of generated questions
        expected_count: Expected number of questions

    Raises:
        ValueError: If questions are invalid or count mismatched
    """
    if not questions:
        _logger.error("No questions were generated")
        raise ValueError("No questions generated")

    if len(questions) != expected_count:
        _logger.warning(f"Expected {expected_count} questions, got {len(questions)}")

    invalid = [q for q in questions if not q.strip() or len(q) < 2]
    if invalid:
        _logger.error(f"Found {len(invalid)} invalid questions: {invalid}")
        raise ValueError(f"Invalid questions generated: {invalid}")


def handle_search_error(error: Exception, query: str) -> dict[str, Any]:
    """
    Handle search errors gracefully.

    Args:
        error: Exception that occurred
        query: Search query that failed

    Returns:
        Error result dictionary
    """
    _logger.error(f"Search failed for query '{query[:80]}...': {error}")

    return {
        "error": str(error),
        "title": "Search Error",
        "url": "",
        "snippet": f"Failed to search: {str(error)[:200]}",
        "engine": "unknown",
    }


def handle_llm_error(error: Exception, operation: str) -> NoReturn:
    """
    Handle LLM errors gracefully.

    Args:
        error: Exception that occurred
        operation: Operation name for logging

    Raises:
        RuntimeError: Always raised
    """
    _logger.error(f"LLM operation '{operation}' failed: {error}")
    raise RuntimeError(f"LLM operation '{operation}' failed after retries: {error}")
