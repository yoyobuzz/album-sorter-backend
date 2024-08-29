from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # JWT Configuration
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 5000

    # MongoDB Configuration
    MONGODB_URI: str
    MONGODB_URI: str = "mongodb+srv://albumsorter:albumsorter@albumsorter.jnncj.mongodb.net/?retryWrites=true&w=majority&appName=albumsorter"
    MONGODB_DB_NAME: str = "albumsorter"

    class Config:
        case_sensitive = True
        # env_file = ".env"  # Optional: Load environment variables from a .env file

# Create the settings instance
settings = Settings()

## TODO make env file

# .env file
# SECRET_KEY=09d25e094faa6ca2556c818166b9f6f0f4caa6cf63b88e8d3e
# ALGORITHM=HS256
# ACCESS_TOKEN_EXPIRE_MINUTES=5000
# MONGODB_URI=your_mongodb_uri
# MONGODB_DB_NAME=albumsorter

# pip install python-dotenv

# from fastapi import FastAPI
# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     SECRET_KEY: str
#     ALGORITHM: str
#     ACCESS_TOKEN_EXPIRE_MINUTES: int
#     MONGODB_URI: str
#     MONGODB_DB_NAME: str

#     class Config:
#         case_sensitive = True
#         env_file = ".env"  # This line is optional if you set environment variables in Vercel

# settings = Settings()

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"SECRET_KEY": settings.SECRET_KEY}
