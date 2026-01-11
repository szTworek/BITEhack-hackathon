"""
Test script for running the pipeline without Celery.
Run from backend directory: python test_pipeline.py
"""
import sys

sys.path.insert(0, ".")

from app.tasks.pipeline import run_pipeline

if __name__ == "__main__":
    print("Starting pipeline manually (without Celery)...\n")
    result = run_pipeline()
    print(f"\nPipeline finished!")
    print(f"Results: {result}")
