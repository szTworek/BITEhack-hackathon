from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "satellite_pipeline",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.pipeline"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # For long-running tasks
)

# Celery Beat schedule - runs at 0:00 and 12:00 UTC
celery_app.conf.beat_schedule = {
    "pipeline-midnight": {
        "task": "app.tasks.pipeline.run_pipeline",
        "schedule": crontab(hour=0, minute=0),
    },
    "pipeline-noon": {
        "task": "app.tasks.pipeline.run_pipeline",
        "schedule": crontab(hour=12, minute=0),
    },
}
