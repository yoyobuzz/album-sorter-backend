from datetime import timedelta, datetime, timezone
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo.collection import Collection
from app.core.config import settings
from app.core.security import verify_password, get_password_hash
from app.schemas import User, UserCreate
from app.core.database import get_db  # Now returns a MongoDB client session
from typing import Annotated
from bson.objectid import ObjectId

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Create an access token with JWT
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Get a user by email from the MongoDB
async def get_user(db: Collection, email: str) -> User:
    user_data = await db["users"].find_one({"email": email})
    if user_data:
        return User(**user_data)
    return None

# Authenticate the user by checking the email and password
async def authenticate_user(db: Collection, email: str, password: str):
    user = await get_user(db, email)
    if not user or not verify_password(password, user.password):
        return False
    return user

# Get the current user by verifying the JWT token
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Collection = Depends(get_db)) -> User:
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

# Create a new user and store it in MongoDB
async def create_user(db: Collection, user: UserCreate) -> User:
    existing_user = await get_user(db, user.email)
    if existing_user:
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
