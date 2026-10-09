from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    phoenix_tracing_enabled: bool = False
    phoenix_collector_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    phoenix_project_name: str = "restaurant-ai-rag"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()