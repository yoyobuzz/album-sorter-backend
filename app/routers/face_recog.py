from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Annotated
from bson import ObjectId
from app.core.database import get_db
from app.repository.auth import get_current_user
from app.schemas import Album, User, Face, FaceCreate, AlbumCreate, ClusterCreate, Cluster
from app.core.security import get_password_hash, verify_password
from app.repository.face_recog import process_urls
from app.repository.face_recog import find_images
router = APIRouter()

@router.get("/albums/{album_id}/find-my-images", response_model=List[str])
async def get_my_photos(
    album_id: str, 
    url: str, 
    user: Annotated[User, Depends(get_current_user)], 
    db = Depends(get_db)
):
    # Find the album and ensure the user has access
    db_album = await db["albums"].find_one({"_id": ObjectId(album_id)})

    # Check if the album exists and the user is in the album's user_ids
    if not db_album or str(user.id) not in db_album["user_ids"]:
        raise HTTPException(status_code=403, detail="You do not have access to this album")

    # Convert the clusters from the database to Pydantic Cluster objects
    clusters = [Cluster(**cluster) for cluster in db_album.get("clusters", [])]

    # Find images similar to the face in the provided URL
    urls = find_images(clusters, url)
    
    if not urls:
        raise HTTPException(status_code=404, detail="No matching photos found in this album")

    # Return the list of image URLs
    return urls