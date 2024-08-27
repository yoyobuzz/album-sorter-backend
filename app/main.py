from fastapi import FastAPI
from app.routers import auth, albums, face_recog
from app.core.database import Base, engine

# Initialize the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include the routes
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(albums.router, prefix="/albums", tags=["albums"])
app.include_router(face_recog.router, prefix="/face-recognition", tags=["face_recognition"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Album Sorter and Find My Photos API!"}
