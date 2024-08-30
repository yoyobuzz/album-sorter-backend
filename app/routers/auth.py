from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.collection import Collection
from typing import Annotated

from app.core.database import get_db
from app.schemas import UserCreate, Token, User
from app.repository.auth import authenticate_user, create_access_token, create_user
from app.repository.auth import get_current_user

router = APIRouter()

# User signup endpoint
@router.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: Collection = Depends(get_db)):
    db_user = await create_user(db, user)
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# User login/token generation endpoint
@router.post("/token", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Collection = Depends(get_db)
):
    db_user = await authenticate_user(db, form_data.username, form_data.password)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# Validate token
@router.get("/valid/", response_model=str)
async def validate(
    user: Annotated[User, Depends(get_current_user)]
):
    return "Valid"