from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables.
    """
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    MONGODB_URI: str
    MONGODB_DB_NAME: str

    class Config:
        case_sensitive = True
        env_file = ".env"

# Initialize settings
settings = Settings()