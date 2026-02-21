"""
Celery configuration for the Radiology Assistant.
"""

import os
from celery import Celery
from .config import Config

# Create the celery instance
app = Celery(
    "radiology_assistant",
    broker=Config.REDIS_URL,
    backend=Config.REDIS_URL,
    include=["radiology_assistant.tasks"]
)

# Optional configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max for any single study orchestration
)

if __name__ == "__main__":
    app.start()
