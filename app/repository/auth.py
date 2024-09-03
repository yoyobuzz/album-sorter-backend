from datetime import timedelta, datetime, timezone
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo.collection import Collection
from app.core.config import settings
from app.core.security import verify_password, get_password_hash
from app.schemas import User, UserCreate
from app.core.database import get_db
from typing import Annotated
from bson.objectid import ObjectId

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Generate a JWT access token with an optional expiration.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_user(db: Collection, email: str) -> User:
    """
    Retrieve a user from the database by email.
    """
    user_data = await db["users"].find_one({"email": email})
    return User(**user_data) if user_data else None

async def authenticate_user(db: Collection, email: str, password: str):
    """
    Verify the user's credentials.
    """
    user = await get_user(db, email)
    if not user or not verify_password(password, user.password):
        return False
    return user

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Collection = Depends(get_db)) -> User:
    """
    Validate the JWT token and return the current user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        user = await get_user(db, email=email)
        if user is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return user

async def create_user(db: Collection, user: UserCreate) -> User:
    """
    Register a new user in the database.
    """
    if await get_user(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    password = get_password_hash(user.password)
    new_user = User(email=user.email, password=password)
    user_dict = new_user.model_dump(by_alias=True, exclude=["id"])
    result = await db["users"].insert_one(user_dict)
    new_user.id = str(result.inserted_id)
    return new_user
