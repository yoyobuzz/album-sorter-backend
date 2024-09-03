from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Annotated
from bson import ObjectId
from app.core.database import get_db
from app.repository.auth import get_current_user
from app.schemas import Album, User, AlbumCreate, AlbumResponse, Cluster
from app.core.security import get_password_hash, verify_password
from app.repository.face_recog import process_urls

router = APIRouter()

@router.get("/user/albums", response_model=List[AlbumResponse])
async def get_user_albums(
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    """
    Fetch all albums associated with the current user.
    """
    user_record = await db["users"].find_one({"_id": ObjectId(user.id)})
    if not user_record or not user_record.get("album_ids"):
        return []
    
    album_ids = [ObjectId(album_id) for album_id in user_record["album_ids"]]
    user_albums = await db["albums"].find({"_id": {"$in": album_ids}}).to_list(length=None)
    return [AlbumResponse(**album) for album in user_albums]

@router.post("/albums", response_model=Album)
async def create_album(
    album: AlbumCreate, 
    current_user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    """
    Create a new album and add it to the user's list of albums.
    """
    hashed_password = get_password_hash(album.password) if album.password else None
    
    album_data = Album(
        album_name=album.album_name,
        user_ids=[str(current_user.id)],
        password=hashed_password
    )
    
    album_dict = album_data.model_dump(by_alias=True, exclude=["id"])
    result = await db["albums"].insert_one(album_dict)
    album_data.id = str(result.inserted_id)
    
    await db["users"].update_one(
        {"_id": ObjectId(current_user.id)},
        {"$push": {"album_ids": album_data.id}}
    )
    return album_data

@router.post("/albums/{album_id}/upload", response_model=List[str])
async def upload_photos(
    album_id: str, 
    photos_urls: List[str], 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    """
    Upload photos to an album and update its clusters.
    """
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id), "user_ids": str(user.id)})
    if not db_album:
        raise HTTPException(status_code=404, detail="Album not found")

    await db["albums"].update_one(
        {"_id": ObjectId(album_id)},
        {"$push": {"image_urls": {"$each": photos_urls}}}
    )

    clusters_old = [Cluster(**cluster) for cluster in db_album.get("clusters", [])]
    clusters_modified = process_urls(photos_urls, clusters_old)

    for modified_cluster in clusters_modified:
        cluster_dict = modified_cluster.model_dump(by_alias=True)

        if modified_cluster.id:
            result = await db["albums"].update_one(
                {"_id": ObjectId(album_id), "clusters._id": ObjectId(modified_cluster.id)},
                {"$set": {"clusters.$": cluster_dict}}
            )
            if result.matched_count == 0:
                await db["albums"].update_one(
                    {"_id": ObjectId(album_id)},
                    {"$push": {"clusters": cluster_dict}}
                )
        else:
            modified_cluster.id = str(ObjectId())
            cluster_dict = modified_cluster.model_dump(by_alias=True)
            await db["albums"].update_one(
                {"_id": ObjectId(album_id)},
                {"$push": {"clusters": cluster_dict}}
            )

    return photos_urls

@router.get("/albums/{album_id}", response_model=List[str])
async def get_album_photos(
    album_id: str, 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    """
    Retrieve all photos in an album by album ID.
    """
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id)})
    
    if not db_album or str(user.id) not in db_album["user_ids"]:
        raise HTTPException(status_code=403, detail="You do not have access to this album")

    return db_album["image_urls"]

@router.post("/albums/{album_id}/add")
async def add_album_to_user(
    album_id: str, 
    password: str, 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    """
    Add an album to the user's list of albums using the album ID and password.
    """
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

@router.delete("/albums/{album_id}/remove")
async def remove_album_from_user(
    album_id: str, 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    """
    Remove an album from the user's list of albums.
    """
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