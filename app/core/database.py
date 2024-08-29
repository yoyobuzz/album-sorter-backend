from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

# Create the MongoDB client
client = AsyncIOMotorClient(settings.MONGODB_URI)

# Access the database
db = client[settings.MONGODB_DB_NAME]

# Dependency to get the database
async def get_db():
    return db