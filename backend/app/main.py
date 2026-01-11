from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.points import router as points_router

app = FastAPI(title="Hackathon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(points_router)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
