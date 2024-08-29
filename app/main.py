from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, albums, face_recog
from app.core.database import get_db
from app.core.config import settings

# Initialize the database tables
# Base.metadata.create_all(bind=engine)

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],  # Allow requests from localhost or React dev server
    allow_credentials=True,  # Allow cookies to be sent in cross-origin requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers, including Authorization
)

# Include the routes
app.include_router(auth.router, prefix="", tags=["auth"])
app.include_router(albums.router, prefix="", tags=["albums"])
app.include_router(face_recog.router, prefix="", tags=["face_recognition"])

@app.get("/")
def read_root():
    return {"Jai Shree Ram!": "Jai Shree Ram!"}