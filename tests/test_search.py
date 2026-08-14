"""Тесты поиска: фильтрация спама, обработка ошибок, загрузка контента."""

from unittest import mock

import pytest
import requests

from minideepresearch.search import (
    ContentFetcher,
    NonRetriableFetchError,
    RetriableFetchError,
    SearXNGSearch,
)


class TestIsValidUrl:
    def test_filters_spam_domains(self):
        search = SearXNGSearch()
        assert not search._is_valid_url("https://www.youtube.com/watch?v=abc")
        assert not search._is_valid_url("https://facebook.com/page")
        assert not search._is_valid_url("https://amazon.com/dp/123")

    def test_allows_normal_domains(self):
        search = SearXNGSearch()
        assert search._is_valid_url("https://docs.python.org/3/library")
        assert search._is_valid_url("https://github.com/example/project")

    def test_rejects_empty_url(self):
        assert not SearXNGSearch()._is_valid_url("")


class TestSearchErrors:
    def test_network_error_returns_empty_list(self):
        search = SearXNGSearch()
        with mock.patch.object(
            requests, "get", side_effect=requests.exceptions.ConnectionError("boom")
        ):
            results = search.search("test query")
        assert results == []

    def test_error_result_does_not_leak_into_sources(self):
        search = SearXNGSearch()
        with mock.patch.object(
            requests, "get", side_effect=requests.exceptions.Timeout("slow")
        ):
            results = search.search("test query")
        assert all(r.get("url") for r in results)


class TestContentFetcher:
    @staticmethod
    def _html_response(html: str, status: int = 200) -> mock.Mock:
        resp = mock.Mock()
        resp.status_code = status
        resp.text = html
        resp.raise_for_status = mock.Mock()
        return resp

    @staticmethod
    def _curl_ok(html: str, code: int = 200) -> mock.Mock:
        proc = mock.Mock()
        proc.returncode = 0
        proc.stderr = ""
        proc.stdout = f"{html}{ContentFetcher._CODE_MARKER}{code}"
        return proc

    def test_fetch_retries_transient_and_raises(self):
        fetcher = ContentFetcher(timeout=1, backend="requests")
        with (
            mock.patch.object(
                requests, "get", side_effect=requests.exceptions.Timeout("boom")
            ) as get_mock,
            mock.patch("time.sleep"),
        ):
            with pytest.raises(RetriableFetchError):
                fetcher.fetch("https://example.com")
        assert get_mock.call_count == 2

    def test_fetch_no_curl_fallback_in_requests_backend(self):
        fetcher = ContentFetcher(timeout=1, backend="requests")
        with (
            mock.patch.object(
                requests, "get", side_effect=requests.exceptions.Timeout("boom")
            ),
            mock.patch("time.sleep"),
            mock.patch("subprocess.run") as run_mock,
        ):
            with pytest.raises(RetriableFetchError):
                fetcher.fetch("https://example.com")
        run_mock.assert_not_called()

    def test_fetch_ssl_verify_error_fails_fast(self):
        fetcher = ContentFetcher(timeout=1, backend="requests")
        ssl_error = requests.exceptions.SSLError(
            "HTTPSConnectionPool(host='x', port=443): Max retries exceeded "
            "(Caused by SSLError(SSLCertVerificationError(1, \
'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: \
certificate has expired (_ssl.c:1032)')))"
        )
        with (
            mock.patch.object(requests, "get", side_effect=ssl_error) as get_mock,
            mock.patch("time.sleep") as sleep_mock,
        ):
            with pytest.raises(NonRetriableFetchError, match="SSL"):
                fetcher.fetch("https://example.com")
        assert get_mock.call_count == 1
        sleep_mock.assert_not_called()

    def test_fetch_403_fails_fast_without_retry(self):
        fetcher = ContentFetcher(timeout=1, backend="requests")
        resp = self._html_response("", status=403)
        with (
            mock.patch.object(requests, "get", return_value=resp) as get_mock,
            mock.patch("time.sleep") as sleep_mock,
        ):
            with pytest.raises(NonRetriableFetchError, match="403"):
                fetcher.fetch("https://example.com")
        assert get_mock.call_count == 1
        sleep_mock.assert_not_called()

    def test_fetch_429_is_retried(self):
        fetcher = ContentFetcher(timeout=1, backend="requests")
        resp = self._html_response("", status=429)
        with (
            mock.patch.object(requests, "get", return_value=resp) as get_mock,
            mock.patch("time.sleep"),
        ):
            with pytest.raises(RetriableFetchError, match="429"):
                fetcher.fetch("https://example.com")
        assert get_mock.call_count == 2

    def test_fetch_rejects_unsupported_scheme(self):
        with pytest.raises(NonRetriableFetchError):
            ContentFetcher().fetch("ftp://example.com/file")

    def test_fetch_uses_browser_user_agent(self):
        fetcher = ContentFetcher(backend="requests")
        resp = self._html_response("<html><body><main>text</main></body></html>")
        with mock.patch.object(requests, "get", return_value=resp) as get_mock:
            fetcher.fetch("https://example.com")
        ua = get_mock.call_args.kwargs["headers"]["User-Agent"]
        assert "Firefox" in ua
        assert "MiniDeepResearch" not in ua

    def test_fetch_extracts_main_content(self):
        fetcher = ContentFetcher(backend="requests")
        html = (
            "<html><body><main><p>Main text</p></main>"
            "<footer>footer junk</footer></body></html>"
        )
        resp = self._html_response(html)
        with mock.patch.object(requests, "get", return_value=resp):
            content = fetcher.fetch("https://example.com")
        assert "Main text" in content
        assert "footer junk" not in content

    def test_fetch_truncates_long_content(self):
        fetcher = ContentFetcher(max_content_chars=100, backend="requests")
        html = f"<html><body><main>{'x' * 500}</main></body></html>"
        resp = self._html_response(html)
        with mock.patch.object(requests, "get", return_value=resp):
            content = fetcher.fetch("https://example.com")
        assert len(content) <= 100 + len("...")

    def test_fetch_multiple_stores_error_message(self):
        fetcher = ContentFetcher()
        with mock.patch.object(
            ContentFetcher, "fetch", side_effect=RuntimeError("boom")
        ):
            results = fetcher.fetch_multiple(["https://example.com"])
        assert results["https://example.com"].startswith("[Error fetching content:")

    def test_fetch_multiple_truncates_multiline_errors(self):
        fetcher = ContentFetcher()
        long_msg = "\n".join(f"line {i}" for i in range(20))
        with mock.patch.object(
            ContentFetcher, "fetch", side_effect=RuntimeError(long_msg)
        ):
            results = fetcher.fetch_multiple(["https://example.com"])
        assert "\n" not in results["https://example.com"]
        assert len(results["https://example.com"]) < 300


class TestCurlFallback:
    """requests (TLS-отпечаток urllib3) → curl (браузерный ClientHello)."""

    @staticmethod
    def _status_response(status: int) -> mock.Mock:
        resp = mock.Mock()
        resp.status_code = status
        resp.text = ""
        resp.raise_for_status = mock.Mock()
        return resp

    def test_403_from_requests_recovered_by_curl(self):
        fetcher = ContentFetcher(timeout=5, backend="auto")
        html = "<html><body><main>PMC article text</main></body></html>"
        with (
            mock.patch.object(requests, "get", return_value=self._status_response(403)),
            mock.patch("subprocess.run", return_value=TestContentFetcher._curl_ok(html)),
        ):
            content = fetcher.fetch("https://pmc.example.com/articles/PMC123")
        assert "PMC article text" in content

    def test_connection_drop_recovered_by_curl(self):
        fetcher = ContentFetcher(timeout=5, backend="auto")
        html = "<html><body><main>chembk data</main></body></html>"
        drop = requests.exceptions.ConnectionError("Remote end closed connection")
        with (
            mock.patch.object(requests, "get", side_effect=drop),
            mock.patch("time.sleep"),
            mock.patch("subprocess.run", return_value=TestContentFetcher._curl_ok(html)),
        ):
            content = fetcher.fetch("https://example.com/chem")
        assert "chembk data" in content

    def test_curl_403_without_mirror_raises(self):
        fetcher = ContentFetcher(timeout=5, backend="auto")
        with (
            mock.patch.object(requests, "get", return_value=self._status_response(403)),
            mock.patch(
                "subprocess.run",
                return_value=TestContentFetcher._curl_ok("", code=403),
            ),
        ):
            with pytest.raises(NonRetriableFetchError, match="403"):
                fetcher.fetch("https://cloudflare-locked.example.com/page")

    def test_curl_unavailable_raises_requests_error(self):
        fetcher = ContentFetcher(timeout=5, backend="auto")
        fetcher._curl_path = None  # curl не установлен в системе
        with (
            mock.patch.object(requests, "get", return_value=self._status_response(403)),
            mock.patch("subprocess.run") as run_mock,
        ):
            with pytest.raises(NonRetriableFetchError, match="403"):
                fetcher.fetch("https://example.com")
        run_mock.assert_not_called()

    def test_curl_ssl_exit_code_raises(self):
        fetcher = ContentFetcher(timeout=5, backend="curl")
        proc = mock.Mock()
        proc.returncode = 60
        proc.stderr = "SSL certificate problem: certificate has expired"
        proc.stdout = ""
        with mock.patch("subprocess.run", return_value=proc):
            with pytest.raises(NonRetriableFetchError, match="curl exit 60"):
                fetcher.fetch("https://expired.example.com")

    def test_curl_backend_skips_requests(self):
        fetcher = ContentFetcher(timeout=5, backend="curl")
        html = "<html><body><main>direct curl</main></body></html>"
        with (
            mock.patch.object(requests, "get") as get_mock,
            mock.patch("subprocess.run", return_value=TestContentFetcher._curl_ok(html)),
        ):
            content = fetcher.fetch("https://example.com")
        get_mock.assert_not_called()
        assert "direct curl" in content

    def test_curl_command_uses_browser_ua_and_timeout(self):
        fetcher = ContentFetcher(timeout=7, backend="curl")
        html = "<html><body><main>text</main></body></html>"
        with mock.patch(
            "subprocess.run", return_value=TestContentFetcher._curl_ok(html)
        ) as run_mock:
            fetcher.fetch("https://example.com")
        cmd = run_mock.call_args.args[0]
        assert "7" in cmd
        assert "Firefox/132.0" in " ".join(cmd)
        assert cmd[-1] == "https://example.com"


class TestMirrorFallback:
    """Зеркала для сайтов, блокирующих любых ботов (pmc → europepmc)."""

    @staticmethod
    def _status_response(status: int) -> mock.Mock:
        resp = mock.Mock()
        resp.status_code = status
        resp.text = ""
        resp.raise_for_status = mock.Mock()
        return resp

    def test_pmc_403_falls_back_to_europepmc(self):
        fetcher = ContentFetcher(timeout=5, backend="auto")

        mirror_html = "<html><body><main>Full article via europepmc</main></body></html>"
        mirror_resp = TestContentFetcher._html_response(mirror_html)

        def requests_side_effect(url, **kwargs):
            if "europepmc.org" in url:
                return mirror_resp
            return self._status_response(403)

        with (
            mock.patch.object(requests, "get", side_effect=requests_side_effect),
            mock.patch(
                "subprocess.run",
                return_value=TestContentFetcher._curl_ok("", code=403),
            ) as run_mock,
        ):
            content = fetcher.fetch("https://pmc.ncbi.nlm.nih.gov/articles/PMC10747162/")

        assert "europepmc" in content
        # curl вызван и для оригинала
        assert any(
            "pmc.ncbi.nlm.nih.gov" in str(c.args[0][-1]) for c in run_mock.call_args_list
        )

    def test_mirror_not_tried_for_ordinary_domains(self):
        fetcher = ContentFetcher(timeout=5, backend="auto")
        with (
            mock.patch.object(requests, "get", return_value=self._status_response(403)),
            mock.patch(
                "subprocess.run",
                return_value=TestContentFetcher._curl_ok("", code=403),
            ) as run_mock,
        ):
            with pytest.raises(NonRetriableFetchError):
                fetcher.fetch("https://sciencedirect.com/science/article/pii/X")
        # Один curl-вызов на оригинал, без второй попытки через зеркало
        assert run_mock.call_count == 1

    def test_mirror_url_format(self):
        mirror = ContentFetcher._mirror_for(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10747162/"
        )
        assert mirror == "https://europepmc.org/article/PMC/PMC10747162"
        assert ContentFetcher._mirror_for("https://example.com/x") is None


class TestRelevanceFilter:
    @staticmethod
    def _search(items: list[dict], query: str) -> list[dict]:
        search = SearXNGSearch()
        resp = mock.Mock()
        resp.status_code = 200
        resp.json = lambda: {"results": items}
        resp.raise_for_status = mock.Mock()
        with mock.patch.object(requests, "get", return_value=resp):
            return search.search(query)

    def test_drops_offtopic_junk(self):
        # Реальный мусор из продового запуска: жд, порно-форумы, автозапчасти
        results = self._search(
            [
                {
                    "url": "https://www.rail.co.il/",
                    "title": "Israel Railways",
                    "content": "train schedule and route planning",
                },
                {
                    "url": "https://lapanchinadimariella.forumfree.it/?t=1",
                    "title": "Racconto erotico",
                    "content": "forum italiano",
                },
                {
                    "url": "https://shop.advanceautoparts.com/",
                    "title": "Advance Auto Parts",
                    "content": "car engine batteries brakes",
                },
                {
                    "url": "https://www.merriam-webster.com/dictionary/advanced",
                    "title": "ADVANCED Definition & Meaning",
                    "content": "definition of advanced",
                },
            ],
            query="kinetic interception methods for hypersonic aerial targets",
        )
        assert results == []

    def test_keeps_relevant_results(self):
        results = self._search(
            [
                {
                    "url": "https://euro-sd.com/articles/x",
                    "title": "Hypersonic weapon interceptor developments",
                    "content": "interception of hypersonic threats",
                },
            ],
            query="kinetic interception methods for hypersonic aerial targets",
        )
        assert len(results) == 1

    def test_stemming_matches_word_forms(self):
        # interception/interceptor, methods/method — совпадение по префиксу
        results = self._search(
            [
                {
                    "url": "https://www.sciencedirect.com/x",
                    "title": "Guidance method for intercepting hypersonic weapons",
                    "content": "",
                },
            ],
            query="interception methods hypersonic",
        )
        assert len(results) == 1

    def test_short_query_requires_single_match(self):
        results = self._search(
            [
                {
                    "url": "https://www.army.mil/x",
                    "title": "PAC-3 missile battery",
                    "content": "",
                },
            ],
            query="PAC-3",
        )
        assert len(results) == 1


class TestHostLimit:
    def test_max_two_results_per_host(self):
        items = [
            {
                "url": f"https://www.mdpi.com/{i}",
                "title": f"Hypersonic interception study part {i}",
                "content": "hypersonic interception methods",
            }
            for i in range(4)
        ]
        search = SearXNGSearch()
        resp = mock.Mock()
        resp.status_code = 200
        resp.json = lambda: {"results": items}
        resp.raise_for_status = mock.Mock()
        with mock.patch.object(requests, "get", return_value=resp):
            results = search.search("hypersonic interception methods")
        assert len(results) == 2
        assert all(r["url"].startswith("https://www.mdpi.com/") for r in results)
