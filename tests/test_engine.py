"""Тесты движка: дефолты фабрики и парсинг атомарных фактов."""

from minideepresearch.engine import MiniResearchEngine, create_engine
from minideepresearch.llm import BaseLLM, LLMResponse


class FakeLLM(BaseLLM):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def invoke(self, prompt: str, *args, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.responses.pop(0), model="fake", raw={})


class FakeSearch:
    def __init__(self, results=None):
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str) -> list[dict]:
        self.queries.append(query)
        return self.results


class FakeProgress:
    def log(self, *args, **kwargs):
        pass

    def show_spinner(self, *args, **kwargs):
        pass

    def stop_spinner(self):
        pass

    def show_progress_bar(self, *args, **kwargs) -> int:
        return 0

    def update_progress_bar(self, *args, **kwargs):
        pass

    def complete_progress_bar(self, *args, **kwargs):
        pass

    def print_result(self, *args, **kwargs):
        pass


class TestCreateEngine:
    def test_num_ctx_default_matches_cli(self):
        engine = create_engine()
        assert engine.llm.num_ctx == 32768

    def test_fact_check_disabled_via_factory(self):
        engine = create_engine(enable_fact_check=False)
        assert engine.synthesizer.enable_fact_check is False

    def test_fact_check_enabled_by_default(self):
        engine = create_engine()
        assert engine.synthesizer.enable_fact_check is True


class TestEvidenceParsing:
    def test_parses_only_well_formed_facts(self):
        llm = FakeLLM(
            [
                "F1: fact one\nF2: fact two\nignored line\nF3x: broken\nF10: fact ten",
                "Итоговый отчёт: текст достаточной длины.",
            ]
        )
        search = FakeSearch(results=[])
        engine = MiniResearchEngine(
            llm=llm,
            search=search,
            strategy="evidence",
            questions_per_iteration=5,
            progress_display=FakeProgress(),
        )

        engine._evidence_based("test query")

        assert search.queries == ["fact one", "fact two", "fact ten"]

    def test_falls_back_to_question_generator_without_facts(self):
        llm = FakeLLM(
            [
                "никаких фактов здесь нет",
                "Q: fallback query",
                "Итоговый отчёт: текст достаточной длины.",
            ]
        )
        search = FakeSearch(results=[])
        engine = MiniResearchEngine(
            llm=llm,
            search=search,
            strategy="evidence",
            questions_per_iteration=3,
            progress_display=FakeProgress(),
        )

        engine._evidence_based("test query")

        assert search.queries == ["fallback query"]
