from pydantic import BaseModel, EmailStr
from typing import List

# Auth Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str


# Album Schemas
class AlbumCreate(BaseModel):
    title: str
    date: str
    password: str

class PhotoUpload(BaseModel):
    url: str

class Album(BaseModel):
    id: int
    title: str
    date: str

class Photo(BaseModel):
    id: int
    url: str

    class Config:
        orm_mode = True
