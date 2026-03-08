"""
Search engine interfaces.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from bs4 import BeautifulSoup

from .logger import get_logger
from .errors import retry, handle_search_error

_logger = get_logger(__name__)


class BaseSearchEngine:
    """Abstract base class for search engines."""

    def search(self, query: str) -> list[dict]:
        """Execute a search query and return results."""
        raise NotImplementedError


class ContentFetcher:
    """Fetches full content from web pages."""

    def __init__(self, timeout: int = 10, max_content_chars: int = 8000):
        self.timeout = timeout
        self.max_content_chars = max_content_chars

    def fetch(self, url: str) -> str:
        """Fetch and extract main content from a URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; MiniDeepResearch/1.0)"
            }
            response = requests.get(
                url, headers=headers, timeout=self.timeout, allow_redirects=True
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Try to find main content
            main = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_=re.compile(r"content|article|post"))
            )

            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n")]
            text = "\n".join(line for line in lines if line)

            # Truncate
            if len(text) > self.max_content_chars:
                text = text[: self.max_content_chars] + "..."

            return text

        except Exception as e:
            return f"[Error fetching content: {e}]"

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def fetch_multiple(
        self, urls: list[str], max_workers: int = 5
    ) -> dict[str, str]:
        """Fetch content from multiple URLs in parallel."""
        results = {}
        
        _logger.debug(f"Fetching {len(urls)} URLs with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.fetch, url): url for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                    _logger.debug(f"Fetched: {url[:80]}...")
                except Exception as e:
                    _logger.warning(f"Failed to fetch {url}: {e}")
                    results[url] = "[Error]"

        return results


class SearXNGSearch(BaseSearchEngine):
    """SearXNG search engine client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 30,
        max_results: int = 10,
        fetch_content: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.fetch_content = fetch_content
        self.fetcher = ContentFetcher() if fetch_content else None

    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def search(self, query: str) -> list[dict[str, Any]]:
        """Search using SearXNG instance."""
        url = f"{self.base_url}/search"

        params = {
            "q": query,
            "format": "json",
            "engines": "general",
            "categories": "general",
            "language": "en",
        }

        _logger.debug(f"Searching SearXNG: {query[:100]}...")
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", [])[: self.max_results]:
                url = item.get("url", "")
                
                # Filter out spammy domains
                if not self._is_valid_url(url):
                    _logger.debug(f"Filtered spam URL: {url}")
                    continue
                
                result = {
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("content", item.get("snippet", "")),
                    "engine": item.get("engine", "searxng"),
                }
                results.append(result)
                
            _logger.debug(f"Found {len(results)} valid search results")

            # Fetch full content if enabled
            if self.fetch_content and self.fetcher and results:
                urls = [r["url"] for r in results if r.get("url")]
                if urls:
                    contents = self.fetcher.fetch_multiple(urls)
                    for r in results:
                        if r.get("url") in contents:
                            r["full_content"] = contents[r["url"]]

            return results

        except requests.exceptions.RequestException as e:
            _logger.error(f"SearXNG search failed: {e}")
            return handle_search_error(e, query)

    def _is_valid_url(self, url: str) -> bool:
        """Filter out spammy or low-quality URLs."""
        if not url:
            return False
        
        # List of domains to filter out (removed Wikipedia and Reddit)
        spam_domains = [
            "amazon.fr", "amazon.de", "amazon.com",
            "zdf.de", "zdfmediathek",
            "facebook.com", "twitter.com", "instagram.com",
            "youtube.com", "tiktok.com",
        ]
        
        url_lower = url.lower()
        
        # Check if URL contains spam domains
        for domain in spam_domains:
            if domain in url_lower:
                return False
        
        return True


class DuckDuckGoSearch(BaseSearchEngine):
    """DuckDuckGo search engine client."""

    def __init__(
        self,
        timeout: int = 30,
        max_results: int = 10,
        fetch_content: bool = False,
    ):
        self.timeout = timeout
        self.max_results = max_results
        self.fetch_content = fetch_content
        self.fetcher = ContentFetcher() if fetch_content else None

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search using DuckDuckGo HTML."""
        url = "https://html.duckduckgo.com/html/"

        data = {"q": query, "b": ""}

        try:
            response = requests.post(url, data=data, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.select(".result")[: self.max_results]:
                title_elem = result.select_one(".result__title")
                link_elem = result.select_one(".result__url")
                snippet_elem = result.select_one(".result__snippet")

                if title_elem:
                    r = {
                        "title": title_elem.get_text(strip=True),
                        "url": link_elem.get_text(strip=True)
                        if link_elem
                        else "",
                        "snippet": snippet_elem.get_text(strip=True)
                        if snippet_elem
                        else "",
                        "engine": "duckduckgo",
                    }
                    results.append(r)

            # Fetch full content if enabled
            if self.fetch_content and self.fetcher and results:
                urls = [r["url"] for r in results if r.get("url")]
                if urls:
                    contents = self.fetcher.fetch_multiple(urls)
                    for r in results:
                        if r.get("url") in contents:
                            r["full_content"] = contents[r["url"]]

            return results

        except Exception as e:
            return [
                {
                    "error": f"DuckDuckGo request failed: {e}",
                    "title": "Error",
                    "url": "",
                    "snippet": "",
                }
            ]
