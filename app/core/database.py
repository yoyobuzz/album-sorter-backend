from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

# Create MongoDB client and access the database
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client[settings.MONGODB_DB_NAME]

async def get_db():
    """
    Dependency that provides a MongoDB database session.
    """
    return db