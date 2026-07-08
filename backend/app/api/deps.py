"""FastAPI dependencies."""

from sqlalchemy.orm import Session
from app.db.session import get_db


def get_db_session():
    """Dependency for getting a database session."""
    yield from get_db()
