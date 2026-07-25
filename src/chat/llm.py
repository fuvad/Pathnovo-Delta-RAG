"""
LLM Client — provider-agnostic interface for Groq (online) and Ollama (offline).

Swappable behind one interface. Reads provider from settings.
"""

from src.config.settings import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified LLM client that routes to Groq or Ollama based on config."""

    def __init__(self):
        settings = get_settings()
        self.provider = settings.LLM_PROVIDER

        if self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            self.model = settings.GROQ_MODEL
        elif self.provider == "ollama":
            import ollama
            self.client = ollama
            self.model = settings.OLLAMA_MODEL
            self.host = settings.OLLAMA_HOST
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        logger.info(
            "llm_client_initialized",
            provider=self.provider,
            model=self.model,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a prompt to the LLM and return the response with metadata.

        Args:
            system_prompt: The system instructions (grounding rules, etc.).
            user_prompt: The user's question with retrieved context.

        Returns:
            Dict with: answer, model, provider, usage (tokens), raw_response.
        """
        if self.provider == "groq":
            return self._call_groq(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt)

    def _call_groq(self, system_prompt: str, user_prompt: str) -> dict:
        """Call Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # low temp for factual, grounded answers
        )

        answer = response.choices[0].message.content
        usage = response.usage

        result = {
            "answer": answer,
            "model": self.model,
            "provider": "groq",
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

        logger.info(
            "llm_response",
            provider="groq",
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )

        return result

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> dict:
        """Call Ollama local API."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.1},
        )

        answer = response["message"]["content"]

        # Ollama token counts
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        result = {
            "answer": answer,
            "model": self.model,
            "provider": "ollama",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        logger.info(
            "llm_response",
            provider="ollama",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return result
