from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://hackathon:hackathon@localhost:5432/hackathon"
    secret_key: str = "dev-secret-key"

    class Config:
        env_file = ".env"


settings = Settings()
