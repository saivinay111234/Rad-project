"""
SQLAlchemy database configuration for the Radiology Assistant.

Provides the engine, session factory, and Base class.
Defaults to SQLite for development; set DATABASE_URL env var for PostgreSQL.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import Config


# Create the database engine
# For SQLite, check_same_thread=False is needed for multi-threaded FastAPI usage
_connect_args = {"check_same_thread": False} if Config.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    Config.DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,  # Validate connections before use (important for PostgreSQL)
)

# Session factory — create a new session per request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


def get_db():
    """
    FastAPI dependency that provides a database session per request.

    Usage:
        @app.get("/endpoint")
        def my_endpoint(db: Session = Depends(get_db)):
            ...

    The session is automatically closed after the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables defined in db_models.py.
    Called on application startup if tables don't exist.
    """
    from . import db_models  # noqa: F401 — import to register ORM models with Base
    Base.metadata.create_all(bind=engine)
