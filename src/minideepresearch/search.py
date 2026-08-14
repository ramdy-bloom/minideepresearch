"""
Search engine interfaces.
"""

import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .logger import get_logger
from .errors import retry, short_error

_logger = get_logger(__name__)

# Стоп-слова для оценки релевантности результатов
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from how in into of on or that the their this
    to using what when where which who why with against under over new best
    """.split()
)


def _stem_terms(text: str) -> set[str]:
    """Извлекает усечённые до 6 символов значимые слова (грубый стемминг)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        w[:6]
        for w in words
        if len(w) >= 3 and not w.isdigit() and w not in _STOPWORDS
    }


class BaseSearchEngine:
    """Abstract base class for search engines."""

    def search(self, query: str) -> list[dict]:
        """Execute a search query and return results."""
        raise NotImplementedError


class FetchError(ValueError):
    """Базовая ошибка загрузки контента."""


class RetriableFetchError(FetchError):
    """Транзиентная ошибка (таймаут, обрыв соединения, 429, 5xx)."""


class NonRetriableFetchError(FetchError):
    """Фатальная ошибка (403/404, невалидный SSL, неподдерживаемый протокол)."""


class ContentFetcher:
    """Fetches full content from web pages.

    Умный фоллбек: requests (браузерные заголовки) → системный curl
    (другой TLS-отпечаток — обходит Akamai-подобные блокировки urllib3)
    → зеркало домена (pmc.ncbi.nlm.nih.gov → europepmc.org).
    """

    # Браузерный UA: с "MiniDeepResearch/1.0" половина сайтов отдаёт 403
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }

    # Маркер для извлечения HTTP-кода из вывода curl (-w)
    _CODE_MARKER = "\n---HTTP_CODE:"

    # Карта зеркал для сайтов, блокирующих не-браузеры
    _MIRRORS = [
        (
            re.compile(r"^https?://pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)/"),
            lambda m: f"https://europepmc.org/article/PMC/{m.group(1)}",
        ),
    ]

    def __init__(
        self,
        timeout: int = 10,
        max_content_chars: int = 8000,
        backend: str = "auto",
    ):
        if backend not in ("auto", "requests", "curl"):
            raise ValueError(f"unknown backend: {backend}")
        self.timeout = timeout
        self.max_content_chars = max_content_chars
        self.backend = backend
        # curl может отсутствовать в системе — фоллбек просто недоступен
        self._curl_path = shutil.which("curl")

    _ALLOWED_SCHEMES = {"http", "https"}

    def fetch(self, url: str) -> str:
        """Fetch and extract main content from a URL.

        Raises:
            FetchError: Когда все бэкенды и зеркала исчерпаны.
        """
        parsed = urlparse(url)
        if parsed.scheme not in self._ALLOWED_SCHEMES:
            raise NonRetriableFetchError(f"unsupported protocol '{parsed.scheme}'")

        if self.backend == "curl":
            return self._fetch_curl(url)

        try:
            return self._requests_with_retry(url)
        except FetchError as e:
            if self.backend == "auto" and self._curl_path:
                _logger.info(
                    f"requests failed ({short_error(e)}), trying curl: {url}"
                )
                return self._fetch_curl(url)
            raise

    @retry(
        max_attempts=2,
        delay=1.0,
        backoff=1.0,
        exceptions=(RetriableFetchError,),
    )
    def _requests_with_retry(self, url: str) -> str:
        return self._fetch_requests_once(url)

    def _fetch_requests_once(self, url: str) -> str:
        try:
            response = requests.get(
                url, headers=self._HEADERS, timeout=self.timeout, allow_redirects=True
            )
        except requests.exceptions.SSLError as e:
            # Просроченный/невалидный сертификат при повторе не изменится
            raise NonRetriableFetchError(
                f"SSL certificate verify failed: {short_error(e)}"
            ) from e
        except requests.exceptions.RequestException as e:
            # Таймауты и обрывы бывают транзиентными — ретраим
            raise RetriableFetchError(short_error(e)) from e

        code = response.status_code
        if code == 429 or code >= 500:
            raise RetriableFetchError(f"HTTP {code} for {url}")
        if code >= 400:
            raise NonRetriableFetchError(f"HTTP {code} for {url}")

        return self._extract_text(response.text)

    def _fetch_curl(self, url: str) -> str:
        """Загрузка системным curl: другой TLS ClientHello, чем у urllib3."""
        if not self._curl_path:
            raise NonRetriableFetchError("curl is not available on this system")

        cmd = [
            self._curl_path,
            "-sL",
            "--compressed",
            "--max-time",
            str(self.timeout),
            "-A",
            self._HEADERS["User-Agent"],
            "-H",
            f"Accept: {self._HEADERS['Accept']}",
            "-H",
            f"Accept-Language: {self._HEADERS['Accept-Language']}",
            "-w",
            f"{self._CODE_MARKER}%{{http_code}}",
            url,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout + 10
            )
        except subprocess.TimeoutExpired as e:
            raise NonRetriableFetchError("curl: subprocess timeout") from e

        body, sep, code_str = proc.stdout.rpartition(self._CODE_MARKER)

        if proc.returncode != 0 and not sep:
            # curl exit 60/35 = SSL, 6 = DNS, 28 = timeout и т.д.
            raise NonRetriableFetchError(
                f"curl exit {proc.returncode}: {short_error(proc.stderr)}"
            )

        if not sep or not code_str.isdigit():
            raise NonRetriableFetchError("curl: no HTTP code in output")

        code = int(code_str)
        if code >= 400:
            mirror = self._mirror_for(url)
            if mirror:
                _logger.info(f"HTTP {code}, trying mirror: {mirror}")
                return self.fetch(mirror)
            raise NonRetriableFetchError(f"HTTP {code} for {url}")

        return self._extract_text(body)

    @classmethod
    def _mirror_for(cls, url: str) -> str | None:
        for pattern, build in cls._MIRRORS:
            m = pattern.match(url)
            if m:
                return build(m)
        return None

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

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

    def fetch_multiple(self, urls: list[str], max_workers: int = 5) -> dict[str, str]:
        """Fetch content from multiple URLs in parallel."""
        results = {}

        _logger.debug(f"Fetching {len(urls)} URLs with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.fetch, url): url for url in urls}

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                    _logger.debug(f"Fetched: {url[:80]}...")
                except Exception as e:
                    _logger.warning(f"Failed to fetch {url}: {short_error(e)}")
                    results[url] = f"[Error fetching content: {short_error(e)}]"

        return results


class SearXNGSearch(BaseSearchEngine):
    """SearXNG search engine client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 30,
        max_results: int = 10,
        fetch_content: bool = False,
        max_results_per_host: int = 2,
        min_query_matches: int = 2,
        fetch_backend: str = "auto",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.fetch_content = fetch_content
        self.max_results_per_host = max_results_per_host
        self.min_query_matches = min_query_matches
        self.fetcher = (
            ContentFetcher(backend=fetch_backend) if fetch_content else None
        )

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

            # Условия релевантности: сколько слов запроса должно встретиться
            # в title/snippet результата (грубый стемминг по префиксу)
            terms = _stem_terms(query)
            required_matches = self.min_query_matches if len(terms) >= 4 else 1

            results = []
            host_counts: dict[str, int] = {}
            for item in data.get("results", [])[: self.max_results]:
                url = item.get("url", "")
                title = item.get("title", "")
                snippet = item.get("content", item.get("snippet", ""))

                # Filter out spammy domains
                if not self._is_valid_url(url):
                    _logger.debug(f"Filtered spam URL: {url}")
                    continue

                # Не более N результатов с одного хоста
                host = urlparse(url).netloc.lower()
                if host_counts.get(host, 0) >= self.max_results_per_host:
                    _logger.debug(f"Skipped {url}: host limit reached")
                    continue

                # Отсекаем нерелевантный мусор (движки SearXNG матчат по
                # одиночным словам: "high-speed" -> жд, "advanced" -> словарь)
                matches = len(_stem_terms(f"{title} {snippet}") & terms)
                if matches < required_matches:
                    _logger.debug(f"Filtered irrelevant result: {title[:60]}")
                    continue

                host_counts[host] = host_counts.get(host, 0) + 1
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "engine": item.get("engine", "searxng"),
                    }
                )

            _logger.debug(
                f"Found {len(results)} valid results "
                f"(filtered by relevance and host limits)"
            )

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
            return []

    def _is_valid_url(self, url: str) -> bool:
        """Filter out spammy or low-quality URLs."""
        if not url:
            return False

        # List of domains to filter out (removed Wikipedia and Reddit)
        spam_domains = [
            "amazon.fr",
            "amazon.de",
            "amazon.com",
            "zdf.de",
            "zdfmediathek",
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "youtube.com",
            "tiktok.com",
        ]

        url_lower = url.lower()

        # Check if URL contains spam domains
        for domain in spam_domains:
            if domain in url_lower:
                return False

        return True
