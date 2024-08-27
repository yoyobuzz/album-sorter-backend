from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_db, get_current_user
from app.models import Album, Photo, User
from app.schemas import AlbumCreate, PhotoUpload, Album as AlbumSchema
from app.core.security import get_password_hash, verify_password
from typing import List

router = APIRouter()

@router.post("/albums", response_model=AlbumSchema)
def create_album(album: AlbumCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db_album = Album(
        title=album.title,
        date=album.date,
        hashed_password=get_password_hash(album.password),
        owner_id=user.id,
    )
    db.add(db_album)
    db.commit()
    db.refresh(db_album)
    return db_album

@router.post("/albums/{album_id}/upload", response_model=PhotoUpload)
def upload_photo(album_id: int, photo: PhotoUpload, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db_album = db.query(Album).filter(Album.id == album_id, Album.owner_id == user.id).first()
    if not db_album:
        raise HTTPException(status_code=404, detail="Album not found")
    
    db_photo = Photo(url=photo.url, album_id=album_id)
    db.add(db_photo)
    db.commit()
    db.refresh(db_photo)
    return db_photo

@router.get("/albums/{album_id}", response_model=List[PhotoUpload])
def get_album_photos(album_id: int, password: str, db: Session = Depends(get_db)):
    db_album = db.query(Album).filter(Album.id == album_id).first()
    if not db_album or not verify_password(password, db_album.hashed_password):
        raise HTTPException(status_code=403, detail="Invalid password")
    
    return db.query(Photo).filter(Photo.album_id == album_id).all()
