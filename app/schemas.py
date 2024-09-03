from pydantic import BaseModel, EmailStr, Field, BeforeValidator
from typing import List, Optional, Annotated
from bson import ObjectId

# Custom type for BSON ObjectId
PyObjectId = Annotated[str, BeforeValidator(str)]

# User Schema
class User(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    email: EmailStr
    password: str
    album_ids: List[str] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

# User Creation Schema
class UserCreate(BaseModel):
    email: EmailStr
    password: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Token Schema
class Token(BaseModel):
    access_token: str
    token_type: str

# Face Schema
class Face(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    embedding: List[float]
    url: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

# Face Creation Schema
class FaceCreate(BaseModel):
    embedding: List[float]
    url: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Cluster Schema
class Cluster(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    cluster_centre: List[float]
    face_images: List[Face] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

# Cluster Creation Schema
class ClusterCreate(BaseModel):
    cluster_centre: List[float]
    face_images: List[Face] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Album Schema
class Album(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    album_name: str
    user_ids: List[str] = []
    image_urls: List[str] = []
    clusters: List[Cluster] = []
    password: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

# Album Response Schema
class AlbumResponse(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    album_name: str
    user_ids: List[str] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        populate_by_name = True

# Album Creation Schema
class AlbumCreate(BaseModel):
    album_name: str
    password: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}