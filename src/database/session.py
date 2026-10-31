import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.database.models import Base

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "traffic_violations.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# Connect arguments needed for SQLite multithreading support with FastAPI/Streamlit
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

def init_db():
    """Create all tables in the configured database."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI and utility dependency to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
