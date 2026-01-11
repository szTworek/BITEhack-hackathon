from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://hackathon:hackathon@localhost:5432/hackathon"
    secret_key: str = "dev-secret-key"

    # Celery / Redis
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # Copernicus API
    copernicus_client_id: str = "sh-7a3e87e6-dc14-4702-b48b-cd24029eb9ec"
    copernicus_client_secret: str = "hR5wvKrrWcZfkFrcmLSC0ZH7UxcuxJk3"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
