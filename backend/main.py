from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import (
    auth_router,
    items_router,
    matches_router,
    notifications_router,
    claims_router,
)
from app.database import connect_db, close_db, get_counts


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="FindIt AI",
    description="Intelligent lost and found network with AI matching and claim verification",
    version="0.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(items_router)
app.include_router(matches_router)
app.include_router(notifications_router)
app.include_router(claims_router)


@app.get("/", tags=["General"])
async def home():
    return {
        "message": "Welcome to FindIt AI!",
        "docs": "/docs",
        "version": "0.6.0",
    }


@app.get("/health", tags=["General"])
async def health_check():
    counts = await get_counts()
    return {
        "status": "healthy",
        "project": "FindIt AI",
        "database": "MongoDB Atlas",
        "reports": counts,
    }
