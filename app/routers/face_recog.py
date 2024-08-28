# from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.core.dependencies import get_db, get_current_user
# from app.models import Photo, Album, User
# from app.repository.face_recog import find_matching_photos

# router = APIRouter()

# @router.post("/albums/{album_id}/find_my_photos")
# def find_my_photos(album_id: int, file: UploadFile = File(...), db: Session = Depends(get_db), user: User = Depends(get_current_user)):
#     db_album = db.query(Album).filter(Album.id == album_id, Album.owner_id == user.id).first()
#     if not db_album:
#         raise HTTPException(status_code=404, detail="Album not found")
    
#     # Process uploaded image and search for matches
#     matching_photos = find_matching_photos(file.file, album_id, db)
    
#     if not matching_photos:
#         raise HTTPException(status_code=404, detail="No matching photos found")
    
#     return matching_photos
