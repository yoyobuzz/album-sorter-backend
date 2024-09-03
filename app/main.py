from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, albums, face_recog

app = FastAPI()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(auth.router, prefix="", tags=["auth"])
app.include_router(albums.router, prefix="", tags=["albums"])
app.include_router(face_recog.router, prefix="", tags=["face_recognition"])

@app.get("/")
def read_root():
    """
    Root endpoint for basic health check.
    """
    return {"message": "Jai Shree Ram!"}
