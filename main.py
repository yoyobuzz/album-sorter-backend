# # # from app.main import app
# # from fastapi import FastAPI

# # app = FastAPI()


# # if __name__ == '__main__':
# #     import uvicorn
# #     uvicorn.run(app)
# #     # uvicorn.run("app", host="0.0.0.0", port=8000, reload=True)

# from fastapi import FastAPI
# from app.routers import auth, albums, face_recog
# from app.core.database import Base, engine

# # Initialize the database tables
# Base.metadata.create_all(bind=engine)

# app = FastAPI()

# # Include the routes
# app.include_router(auth.router, prefix="/auth", tags=["auth"])
# app.include_router(albums.router, prefix="/albums", tags=["albums"])
# # app.include_router(face_recog.router, prefix="/face-recognition", tags=["face_recognition"])

# @app.get("/")
# def read_root():
#     return {"###": "Jai Shree Ram!"}

from app.main import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app", host="0.0.0.0", port=8000, reload=True)