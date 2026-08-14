"""Тесты синтеза: full_content в промпте и парсинг факт-чека."""

import pytest

from minideepresearch.llm import BaseLLM, LLMResponse
from minideepresearch.synthesis import SynthesisHandler


class FakeLLM(BaseLLM):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str, *args, **kwargs) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content=self.responses.pop(0), model="fake", raw={})


class TestFormatSources:
    def test_prefers_full_content_over_snippet(self):
        handler = SynthesisHandler(FakeLLM([]))
        formatted = handler._format_sources(
            [
                {
                    "title": "Title",
                    "url": "https://a.example",
                    "snippet": "short snippet",
                    "full_content": "LONG FULL PAGE TEXT",
                }
            ]
        )
        assert "LONG FULL PAGE TEXT" in formatted
        assert "short snippet" not in formatted

    def test_falls_back_to_snippet_without_full_content(self):
        handler = SynthesisHandler(FakeLLM([]))
        formatted = handler._format_sources(
            [{"title": "Title", "url": "https://a.example", "snippet": "snippet"}]
        )
        assert "snippet" in formatted

    def test_skips_error_full_content(self):
        handler = SynthesisHandler(FakeLLM([]))
        formatted = handler._format_sources(
            [
                {
                    "title": "Title",
                    "url": "https://a.example",
                    "snippet": "good snippet",
                    "full_content": "[Error fetching content: boom]",
                }
            ]
        )
        assert "good snippet" in formatted
        assert "boom" not in formatted

    def test_source_without_any_content_keeps_url(self):
        handler = SynthesisHandler(FakeLLM([]))
        formatted = handler._format_sources([{"title": "Title", "url": "https://a"}])
        assert "[1] Title" in formatted
        assert "https://a" in formatted


class TestFactCheck:
    def _run(self, fact_check_response: str) -> dict:
        llm = FakeLLM(
            [fact_check_response, "Итоговый отчёт: достаточно длинный текст ответа."]
        )
        handler = SynthesisHandler(llm, enable_fact_check=True)
        return handler.synthesize(
            "query",
            [{"title": "T", "url": "https://a", "snippet": "s"}],
            previous_knowledge="previous knowledge",
        )

    @pytest.mark.parametrize(
        "response",
        [
            "CONSISTENT",
            "consistent",
            "Consistent with previous findings",
            "Противоречий нет",
            "Согласовано с предыдущими выводами",
        ],
    )
    def test_consistent_variants_produce_no_note(self, response):
        result = self._run(response)
        assert result["fact_check"] is None

    def test_conflict_returns_note(self):
        note = "CONFLICT: источник [2] противоречит предыдущим выводам"
        result = self._run(note)
        assert result["fact_check"] == note

    def test_empty_fact_check_response_produces_no_note(self):
        result = self._run("   ")
        assert result["fact_check"] is None

    def test_fact_check_skipped_without_previous_knowledge(self):
        llm = FakeLLM(["Полный ответ синтеза, длиннее десяти символов."])
        handler = SynthesisHandler(llm, enable_fact_check=True)
        handler.synthesize("query", [{"title": "T", "url": "https://a", "snippet": "s"}])
        assert len(llm.prompts) == 1

    def test_prompt_forbids_own_sources_section(self):
        llm = FakeLLM(["Полный ответ синтеза, длиннее десяти символов."])
        handler = SynthesisHandler(llm)
        handler.synthesize("query", [{"title": "T", "url": "https://a", "snippet": "s"}])
        assert "Do NOT include" in llm.prompts[0]
