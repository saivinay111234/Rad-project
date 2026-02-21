"""
LLM Client wrapper supporting Gemini and Ollama providers.

Provides a clean interface to interact with LLMs, with:
- PHI scrubbing before prompt dispatch (privacy layer)
- Automatic retry with exponential backoff
- Circuit breaker to prevent cascading failures
- Pluggable Ollama backend for on-premise deployment
"""

import json
import logging
import time
from typing import Optional, Dict, Any
import requests

from .config import Config
from .phi_scrubber import PHIScrubber, get_phi_scrubber

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client supporting Gemini (cloud) and Ollama (on-premise) backends.
    
    Features:
    - PHI scrubbing before any prompt leaves the local environment
    - Automatic retry with exponential backoff
    - Circuit breaker: opens after `circuit_failure_threshold` consecutive
      failures, resets after `circuit_reset_seconds` seconds
    """

    # Circuit breaker state (class-level so all instances share it)
    _circuit_failures: int = 0
    _circuit_open: bool = False
    _circuit_last_failure: float = 0.0
    CIRCUIT_FAILURE_THRESHOLD: int = 3
    CIRCUIT_RESET_SECONDS: int = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        provider: Optional[str] = None,
        phi_scrubber: Optional[PHIScrubber] = None,
        scrub_phi: bool = True,
    ):
        """
        Initialize LLM client.

        Args:
            api_key: Gemini API key (uses Config.GEMINI_API_KEY if not provided)
            temperature: Sampling temperature (0.0-1.0)
            provider: "gemini" or "ollama". Defaults to Config.LLM_PROVIDER.
            phi_scrubber: Custom PHIScrubber instance. Defaults to singleton.
            scrub_phi: Whether to scrub PHI from prompts before sending. Default True.
        """
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self.temperature = temperature
        self.scrub_phi = scrub_phi
        self.phi_scrubber = phi_scrubber if phi_scrubber is not None else get_phi_scrubber()

        if self.provider == "ollama":
            self.model = Config.OLLAMA_MODEL
            self.base_url = Config.OLLAMA_BASE_URL
            self.api_key = None  # Ollama doesn't need a key
        else:
            # Default: Gemini
            self.provider = "gemini"
            self.api_key = api_key if api_key is not None else Config.GEMINI_API_KEY
            self.model = "gemini-2.5-flash"
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
            if not self.api_key:
                raise ValueError("Gemini API key is required when using the gemini provider")

        logger.info("LLMClient initialized: provider=%s model=%s scrub_phi=%s",
                    self.provider, self.model, self.scrub_phi)
    
    def _check_circuit(self) -> None:
        """Raise if circuit breaker is open and hasn't reset yet."""
        if LLMClient._circuit_open:
            elapsed = time.time() - LLMClient._circuit_last_failure
            if elapsed < LLMClient.CIRCUIT_RESET_SECONDS:
                raise RuntimeError(
                    f"LLM circuit breaker is open (resets in "
                    f"{LLMClient.CIRCUIT_RESET_SECONDS - int(elapsed)}s). "
                    "Too many consecutive LLM failures."
                )
            # Auto-reset after timeout
            logger.info("Circuit breaker auto-reset after %ds", int(elapsed))
            LLMClient._circuit_open = False
            LLMClient._circuit_failures = 0

    def _record_success(self) -> None:
        LLMClient._circuit_failures = 0
        LLMClient._circuit_open = False

    def _record_failure(self) -> None:
        LLMClient._circuit_failures += 1
        LLMClient._circuit_last_failure = time.time()
        if LLMClient._circuit_failures >= LLMClient.CIRCUIT_FAILURE_THRESHOLD:
            LLMClient._circuit_open = True
            logger.error("Circuit breaker OPENED after %d consecutive failures", LLMClient._circuit_failures)

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text using the configured LLM provider (Gemini or Ollama).

        Args:
            prompt: The prompt to send to the model
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt/instructions

        Returns:
            Generated text response

        Raises:
            ValueError: If API returns an error
            RuntimeError: If request fails after retries or circuit is open
        """
        # 1. Check circuit breaker
        self._check_circuit()

        # 2. PHI scrubbing — sanitise prompt before it leaves the system
        if self.scrub_phi:
            prompt = self.phi_scrubber.scrub(prompt)
            if system_prompt:
                system_prompt = self.phi_scrubber.scrub(system_prompt)

        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or Config.LLM_MAX_TOKENS

        # 3. Dispatch to correct provider
        try:
            if self.provider == "ollama":
                result = self._generate_ollama(prompt, temp, max_tokens, system_prompt)
            else:
                result = self._generate_gemini(prompt, temp, max_tokens, system_prompt)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    def _generate_gemini(
        self,
        prompt: str,
        temp: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> str:
        """Internal: call Gemini API with retry logic."""
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        retries = 0
        last_error = None

        while retries < Config.MAX_RETRIES:
            try:
                response = requests.post(
                    f"{self.base_url}/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    response_json = response.json()
                    if "candidates" in response_json and len(response_json["candidates"]) > 0:
                        candidate = response_json["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            text = candidate["content"]["parts"][0]["text"]
                            logger.info("Gemini generation successful (retries=%d)", retries)
                            return text
                    last_error = "Unexpected response format from Gemini API"
                    retries += 1
                elif response.status_code == 429:
                    last_error = "Rate limited by Gemini API"
                    retries += 1
                    if retries < Config.MAX_RETRIES:
                        wait_time = Config.RETRY_DELAY * (2 ** retries)
                        logger.warning("Rate limited. Retrying in %ss... (attempt %d)", wait_time, retries)
                        time.sleep(wait_time)
                else:
                    error_msg = f"Gemini API error {response.status_code}: {response.text}"
                    raise ValueError(error_msg)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                retries += 1
                if retries < Config.MAX_RETRIES:
                    logger.warning("Request failed, retrying... (attempt %d): %s", retries, e)
                    time.sleep(Config.RETRY_DELAY)

        raise RuntimeError(f"Gemini: failed after {Config.MAX_RETRIES} attempts: {last_error}")

    def _generate_ollama(
        self,
        prompt: str,
        temp: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> str:
        """Internal: call Ollama local API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens,
            },
        }

        retries = 0
        last_error = None

        while retries < Config.MAX_RETRIES:
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=120,  # Local inference can be slower
                )
                if response.status_code == 200:
                    data = response.json()
                    text = data.get("message", {}).get("content", "")
                    if text:
                        logger.info("Ollama generation successful (model=%s retries=%d)", self.model, retries)
                        return text
                    last_error = "Empty response from Ollama"
                    retries += 1
                else:
                    raise ValueError(f"Ollama API error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                retries += 1
                if retries < Config.MAX_RETRIES:
                    logger.warning("Ollama request failed, retrying (attempt %d): %s", retries, e)
                    time.sleep(Config.RETRY_DELAY)

        raise RuntimeError(f"Ollama: failed after {Config.MAX_RETRIES} attempts: {last_error}")

    
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON response from the model.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Override default temperature
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response
            
        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        response_text = self.generate(prompt, temperature, system_prompt=system_prompt)
        
        # Try to extract JSON from response (model might add extra text)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise
