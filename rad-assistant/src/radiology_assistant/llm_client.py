"""
LLM Client wrapper for Gemini API.

Provides a clean interface to interact with the Gemini model.
"""

import json
import logging
from typing import Optional, Dict, Any
import time
import requests

from .config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with Gemini LLM."""
    
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize LLM client.
        
        Args:
            api_key: Gemini API key (uses Config.GEMINI_API_KEY if not provided)
            temperature: Sampling temperature (0.0-1.0)
        """
        # If api_key is provided explicitly (even empty), honor it; otherwise use Config
        self.api_key = api_key if api_key is not None else Config.GEMINI_API_KEY
        self.temperature = temperature
        self.model = "gemini-2.5-flash"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional system prompt/instructions
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If API returns an error
            RuntimeError: If request fails after retries
        """
        temp = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or Config.LLM_MAX_TOKENS
        
        # Build the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tokens,
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": system_prompt
                    }
                ]
            }
        
        # Retry logic
        retries = 0
        last_error = None
        
        while retries < Config.MAX_RETRIES:
            try:
                response = requests.post(
                    f"{self.base_url}/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_json = response.json()

                    # Extract text from response
                    if "candidates" in response_json and len(response_json["candidates"]) > 0:
                        candidate = response_json["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            text = candidate["content"]["parts"][0]["text"]
                            logger.info(f"LLM generation successful (retries: {retries})")
                            return text

                    # Treat unexpected response format as retriable
                    last_error = "Unexpected response format from Gemini API"
                    retries += 1
                    if retries < Config.MAX_RETRIES:
                        wait_time = Config.RETRY_DELAY * (2 ** retries)
                        logger.warning(f"Unexpected response format. Retrying in {wait_time}s... (attempt {retries})")
                        time.sleep(wait_time)
                        continue
                    # will fall through to raise after loop
                
                elif response.status_code == 429:
                    # Rate limited - back off and retry
                    last_error = "Rate limited by API"
                    retries += 1
                    if retries < Config.MAX_RETRIES:
                        wait_time = Config.RETRY_DELAY * (2 ** retries)  # Exponential backoff
                        logger.warning(f"Rate limited. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                
                else:
                    error_msg = f"Gemini API error {response.status_code}: {response.text}"
                    raise ValueError(error_msg)
                    
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                retries += 1
                if retries < Config.MAX_RETRIES:
                    logger.warning(f"Request failed. Retrying... (attempt {retries})")
                    time.sleep(Config.RETRY_DELAY)
        
        # Failed after all retries
        raise RuntimeError(f"Failed to generate content from LLM after {Config.MAX_RETRIES} attempts: {last_error}")
    
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
