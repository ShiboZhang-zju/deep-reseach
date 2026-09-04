"""Database engine and session factory."""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 10},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable WAL mode and busy_timeout for concurrent access.

    busy_timeout is 10s, NOT the SQLAlchemy default 30s+: agent sessions run
    synchronous SQLAlchemy calls directly on the event loop thread, so a
    single lock wait freezes ALL HTTP traffic for that duration. Under 3-way
    task concurrency a 30s wait cascaded into ~10 minutes of unresponsive API
    (multiple queued waits back-to-back). 10s is still 10x the longest healthy
    write transaction (per-query commits keep writes sub-second); on timeout
    the existing per-phase retries with backoff absorb the failure instead of
    the whole API going dark.
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    cursor.close()


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
