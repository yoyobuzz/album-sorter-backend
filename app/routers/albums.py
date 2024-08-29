from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Annotated
from bson import ObjectId
from app.core.database import get_db
from app.repository.auth import get_current_user
from app.schemas import Album, User, Image, ImageCreate, AlbumCreate
from app.core.security import get_password_hash, verify_password

router = APIRouter()

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from bson import ObjectId
from app.core.database import get_db
from app.repository.auth import get_current_user
from app.schemas import Album, User, AlbumCreate
from app.core.security import get_password_hash

router = APIRouter()

# Create a new album and add it to the user's list of albums
@router.post("/albums", response_model=Album)
async def create_album(
    album: AlbumCreate, 
    current_user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    # Hash the password before saving it to the database
    hashed_password = get_password_hash(album.password) if album.password else None
    
    # Create the album data
    album_data = Album(
        album_name=album.album_name,
        user_ids=[str(current_user.id)],  # Automatically add the current user's ID
        image_urls=album.image_urls,
        clusters=album.clusters,
        password=hashed_password
    )
    
    # Insert the album into the database
    album_dict = album_data.model_dump(by_alias=True, exclude=["id"])  # Use by_alias=True to handle _id properly
    result = await db["albums"].insert_one(album_dict)
    album_data.id = str(result.inserted_id)
    
    # Add the album ID to the current user's list of albums
    await db["users"].update_one(
        {"_id": ObjectId(current_user.id)},
        {"$push": {"album_ids": album_data.id}}
    )
    return album_data

# Upload multiple photos to an album
@router.post("/albums/{album_id}/upload", response_model=List[str])
async def upload_photos(
    album_id: str, 
    photos_urls: List[str], 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id), "user_ids": str(user.id)})
    if not db_album:
        raise HTTPException(status_code=404, detail="Album not found")

    # photo_urls = [photo.url for photo in photos]
    await db["albums"].update_one(
        {"_id": ObjectId(album_id)},
        {"$push": {"image_urls": {"$each": photos_urls}}}
    )
    return photos_urls

# Get all photos in an album by album ID
@router.get("/albums/{album_id}", response_model=List[str])
async def get_album_photos(
    album_id: str, 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    # Find the album by ID
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id)})
    
    # Check if the album exists and the user is in the album's user_ids
    if not db_album or str(user.id) not in db_album["user_ids"]:
        raise HTTPException(status_code=403, detail="You do not have access to this album")

    # Return the list of image URLs
    return db_album["image_urls"]

# Add an album to the user's list of albums using album ID and password
@router.post("/albums/{album_id}/add")
async def add_album_to_user(
    album_id: str, 
    password: str,  # Password now comes in the request body
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id)})
    if not db_album or not verify_password(password, db_album.get('password')):
        raise HTTPException(status_code=403, detail="Invalid password")
    
    if str(user.id) not in db_album["user_ids"]:
        await db["albums"].update_one(
            {"_id": ObjectId(album_id)},
            {"$push": {"user_ids": str(user.id)}}
        )
        await db["users"].update_one(
            {"_id": ObjectId(user.id)},
            {"$push": {"album_ids": album_id}}
        )

    return {"msg": "Album added to your list"}

# Get all albums associated with the current user
@router.get("/user/albums", response_model=List[Album])
async def get_user_albums(
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    user_albums = await db["albums"].find({"user_ids": str(user.id)}).to_list(length=None)
    return user_albums

# Remove an album from the user's list of albums
@router.delete("/albums/{album_id}/remove")
async def remove_album_from_user(
    album_id: str, 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id), "user_ids": str(user.id)})
    if not db_album:
        raise HTTPException(status_code=404, detail="Album not found or not in your list")

    await db["albums"].update_one(
        {"_id": ObjectId(album_id)},
        {"$pull": {"user_ids": str(user.id)}}
    )
    await db["users"].update_one(
        {"_id": ObjectId(user.id)},
        {"$pull": {"album_ids": album_id}}
    )
    return {"msg": "Album removed from your list"}
