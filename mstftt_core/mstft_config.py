# mstftt_core/mstft_config.py

import os
import logging


def get_client_config():
    """
    Returns a configuration dictionary for the MSTFT demo.
    This includes API settings, LLM default parameters, and logging configuration.
    
    Environment Variables:
      - DEFAULT_MODEL_CHECKPOINT: e.g. "gpt-4o" (default if not set)
      - DEFAULT_TEMPERATURE: e.g. 0.8
      - DEFAULT_TOP_P: e.g. 0.9
      - DEFAULT_PRESENCE_PENALTY: e.g. 0.8
      - DEFAULT_FREQUENCY_PENALTY: e.g. 0.3
      - DEFAULT_MAX_TOKENS: e.g. 1024
      - LOG_LEVEL: e.g. "INFO"
      - LOG_FILE: e.g. "mstft_demo.log"
      - API_TIMEOUT: e.g. 60 (seconds)
    
    Returns:
        dict: Configuration settings for MSTFT.
    """
    default_llm_settings = {
        "model_checkpoint": os.getenv("DEFAULT_MODEL_CHECKPOINT", "gpt-4o"),
        "temperature": float(os.getenv("DEFAULT_TEMPERATURE", 0.8)),
        "top_p": float(os.getenv("DEFAULT_TOP_P", 0.9)),
        "presence_penalty": float(os.getenv("DEFAULT_PRESENCE_PENALTY", 0.8)),
        "frequency_penalty": float(os.getenv("DEFAULT_FREQUENCY_PENALTY", 0.3)),
        "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", 1024))
    }

    logging_config = {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": os.getenv("LOG_FILE", "mstft_demo.log")
    }
    logging.basicConfig(
        level=getattr(logging, logging_config["level"]),
        filename=logging_config["log_file"],
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = {
        "api_version": "1.0",
        "client_name": "MSTFT Demo Client",
        "llm_settings": default_llm_settings,
        "timeout": int(os.getenv("API_TIMEOUT", 60)),
        "logging": logging_config
    }
    return config
