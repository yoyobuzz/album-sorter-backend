from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # JWT Configuration
    SECRET_KEY: str = "your-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database Configuration
    SQLALCHEMY_DATABASE_URL: str = "sqlite:///./test.db"
    
    # S3 or Cloudinary settings
    CLOUDINARY_URL: str = "your-cloudinary-url"

    class Config:
        case_sensitive = True
        env_file = ".env"  # Optional: Load environment variables from a .env file

# Create the settings instance
settings = Settings()
