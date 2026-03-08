"""
MiniDeepResearch - Minimal deep research engine for small LLM models

Usage:
    from minideepresearch import MiniResearchEngine, setup_logger

    engine = MiniResearchEngine(
        llm=OllamaLLM(model="llama2:7b"),
        search=SearXNGSearch(base_url="http://localhost:8080"),
        strategy="multi_iter",
        max_iterations=3,
    )

    result = engine.research("How does CRISPR work?")
    print(result["answer"])
"""

from .engine import MiniResearchEngine, create_engine
from .llm import OllamaLLM
from .logger import setup_logger
from .search import SearXNGSearch, BaseSearchEngine, ContentFetcher

__version__ = "0.1.0"

__all__ = [
    "MiniResearchEngine",
    "create_engine",
    "OllamaLLM",
    "SearXNGSearch",
    "BaseSearchEngine",
    "ContentFetcher",
    "setup_logger",
]
