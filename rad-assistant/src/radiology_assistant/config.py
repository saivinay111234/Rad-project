"""
Configuration management for the radiology assistant.

Handles environment variables and settings.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # LLM Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "ollama"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Retry Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))

    # Security / JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./rad_assistant.db")

    # Redis / Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configurations are set."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        return True
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Return configuration as a dictionary."""
        return {
            "gemini_api_key": "***" if cls.GEMINI_API_KEY else None,
            "llm_temperature": cls.LLM_TEMPERATURE,
            "llm_max_tokens": cls.LLM_MAX_TOKENS,
            "llm_provider": cls.LLM_PROVIDER,
            "ollama_model": cls.OLLAMA_MODEL,
            "debug": cls.DEBUG,
            "log_level": cls.LOG_LEVEL,
            "database_url": cls.DATABASE_URL,
            "redis_url": cls.REDIS_URL,
        }

    # Agent 6: Worklist Triage Configuration
    # We define a default configuration here that can be used if no external config is provided.
    # In a real app, this might be loaded from a YAML/JSON file or DB.
    # Local imports to avoid circular dependency at module level if models imports config
    @classmethod
    def get_triage_config(cls) -> 'TriageConfig':
        """Load triage configuration from yaml file or fallback to defaults."""
        import yaml
        from .models import TriageConfig, TriageThresholdConfig, ModalityGroup
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "triage_config.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return TriageConfig(**data)
            except Exception:
                # Fallback to defaults on parse error
                pass

        # Default fallback
        return TriageConfig(
            thresholds=[
                TriageThresholdConfig(
                    modality_group=ModalityGroup.XR,
                    body_region="Chest",
                    critical_threshold=0.9,
                    high_threshold=0.7,
                    low_threshold=0.3
                )
            ],
            max_batch_size=10,
            enable_llm_explanation=True,
            model_mapping={
                "XR/Chest": "densenet121-res224-all"
            }
        )
