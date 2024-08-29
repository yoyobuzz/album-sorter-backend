from pydantic import BaseModel, EmailStr, Field, BeforeValidator
from typing import List, Optional, Annotated
from bson import ObjectId

PyObjectId = Annotated[str, BeforeValidator(str)]

# Auth Schemas
class User(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    email: EmailStr
    password: str
    album_ids: List[str] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name  = True  # This allows using 'id' instead of '_id'

class UserCreate(BaseModel):
    email: EmailStr
    password: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

# Image Schemas
class Face(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    embedding: List[float]  # Vector of 128 floats
    url: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name  = True

class FaceCreate(BaseModel):
    embedding: List[float]
    url: str
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Cluster Schemas
class Cluster(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    cluster_centre: List[float]  # Vector of 128 floats
    face_images: List[Face] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name  = True

class ClusterCreate(BaseModel):   
    cluster_centre: List[float]  # Vector of 128 floats
    face_images: List[Face] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Album Schemas
class Album(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    album_name: str  # Name of the album
    user_ids: List[str] = []  # List of user IDs associated with the album
    image_urls: List[str] = []  # List of URLs of all images in the album
    clusters: List[Cluster] = []
    password: str  # Hashed password for the album

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

class AlbumCreate(BaseModel):
    album_name: str  # Name of the album
    password: str  # Plaintext password for the album (will be hashed)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}