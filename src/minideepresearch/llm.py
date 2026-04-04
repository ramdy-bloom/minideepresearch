"""
LLM interface for Ollama (local models).
"""

import json
from abc import ABC, abstractmethod
from typing import Any

import requests

from .logger import get_logger
from .errors import retry, validate_llm_response, handle_llm_error

_logger = get_logger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def invoke(self, prompt: str) -> "LLMResponse":
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    """Ollama API client for local LLM inference."""

    def __init__(
        self,
        model: str = "llama2:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: int = 300,
        num_predict: int = -1,  # Максимальное количество токенов (-1 = неограниченно)
        use_mmap: bool = True,  # Использовать mmap для загрузки модели
        num_ctx: int = 4096,  # Размер контекста
        enable_thinking: bool = True,  # Включить "размышления" модели (для Qwen3.5)
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.num_predict = num_predict
        self.use_mmap = use_mmap
        self.num_ctx = num_ctx
        self.enable_thinking = enable_thinking

    @retry(max_attempts=3, delay=2.0, backoff=1.5)
    def invoke(
        self, prompt: str, num_predict: int = -1, min_length: int = 10
    ) -> "LLMResponse":
        """Send a prompt to Ollama and get the response.

        Args:
            prompt: Текст запроса
            num_predict: Максимальное количество генерируемых токенов (-1 = неограниченно)
                        Уменьшение этого значения помогает избежать 'размышлений' модели
            min_length: Минимально допустимая длина ответа
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "num_predict": num_predict if num_predict != -1 else None,
            "options": {
                "num_ctx": self.num_ctx,
                "use_mmap": self.use_mmap,
            },
        }

        # Для моделей Qwen3.5 добавляем chat_template_kwargs для отключения thinking
        if "qwen" in self.model.lower() and not self.enable_thinking:
            payload["options"] = payload.get("options", {})
            payload["options"]["chat_template_kwargs"] = {"enable_thinking": False}

        _logger.debug(f"Sending LLM request (model={self.model}, len={len(prompt)})")

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            content = data.get("response", "")

            # Validate response
            validate_llm_response(content, self.model, min_length=min_length)

            _logger.debug(f"LLM response received: {len(content)} chars")

            return LLMResponse(
                content=content,
                model=self.model,
                raw=data,
            )
        except requests.exceptions.RequestException as e:
            handle_llm_error(e, "invoke")

    @retry(max_attempts=3, delay=2.0, backoff=1.5)
    def invoke_with_messages(
        self, messages: list[dict], num_predict: int = -1, min_length: int = 10
    ) -> "LLMResponse":
        """Send a chat-style prompt to Ollama (chat API).

        Args:
            messages: Список сообщений в формате chat API
            num_predict: Максимальное количество генерируемых токенов
                        Меньше значение = меньше 'размышлений' модели
            min_length: Минимально допустимая длина ответа
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "num_predict": num_predict if num_predict != -1 else None,
            "options": {
                "num_ctx": self.num_ctx,
                "use_mmap": self.use_mmap,
            },
        }

        # Для моделей Qwen3.5 добавляем chat_template_kwargs для отключения thinking
        if "qwen" in self.model.lower() and not self.enable_thinking:
            payload["options"] = payload.get("options", {})
            payload["options"]["chat_template_kwargs"] = {"enable_thinking": False}

        _logger.debug(
            f"Sending LLM chat request (model={self.model}, len={len(str(messages))})"
        )

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")

            # Validate response
            validate_llm_response(content, self.model, min_length=min_length)

            _logger.debug(f"LLM chat response received: {len(content)} chars")

            return LLMResponse(
                content=content,
                model=self.model,
                raw=data,
            )
        except requests.exceptions.RequestException as e:
            handle_llm_error(e, "invoke_with_messages")

    def __repr__(self):
        return f"OllamaLLM(model={self.model!r}, base_url={self.base_url!r})"


class LLMResponse:
    """Wrapper for LLM response."""

    def __init__(self, content: str, model: str, raw: dict):
        self.content = content
        self.model = model
        self.raw = raw

    def __str__(self):
        return self.content


class LLMError(Exception):
    """Exception for LLM errors."""

    pass
