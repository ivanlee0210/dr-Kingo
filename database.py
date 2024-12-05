## Database module for dr Kingo
## Class HAI5016 - by camphouse.me

from loguru import logger

from dotenv import load_dotenv
import os

from sqlalchemy import create_engine, Column, Integer, String, MetaData, DateTime, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

logger.add("logs/database.log", rotation="10 MB")

# Define the database connection string
load_dotenv()
connection_string = os.getenv('DATABASE_URL')

def get_engine():
    if not connection_string:
        logger.error("SUPABASE_CONNECTION_STRING environment variable is not set.")
        raise ValueError("SUPABASE_CONNECTION_STRING environment variable is not set.")
    else:
        logger.info("Database connection string loaded successfully.")

    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        logger.info("SQLAlchemy engine created successfully.")
        return engine
    except Exception as e:
        logger.error(f"An error occurred while creating the engine: {e}")
        raise

def get_session(engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

metadata = MetaData(schema='public')
Base = declarative_base(metadata=metadata)

class skkuUrls(Base):
    __tablename__ = 'skku_urls'

    campus: int = Column(Integer)
    info: str = Column(String)
    url: str = Column(String, primary_key=True, unique=True, index=True)
    last_download: datetime = Column(DateTime)

class skkuHtml(Base):
    __tablename__ = 'skku_html'

    url: str = Column(String, primary_key=True)
    html: str = Column(String)
    scraping_date: datetime = Column(DateTime, default=datetime.utcnow)
    cont_wrap_hash: str = Column(String)
    campus: int = Column(Integer, nullable=True)

class skkuMd(Base):
    __tablename__ = 'skku_md'

    url: str = Column(String, primary_key=True)
    markdown: str = Column(String)
    total_tokens: int = Column(Integer)
    model: str = Column(String)
    usage: str = Column(String)
    html_wrap_hash: str = Column(String)
    md_wrap_hash: str = Column(String)
    created: int = Column(BigInteger)
    generation_date: datetime = Column(DateTime, default=datetime.utcnow)

def create_tables():
    try:
        # Create a SQLAlchemy engine
        engine = get_engine()

        # Define the table schemas using SQLAlchemy
        metadata = MetaData(schema='public')

        # Use imported table schemas
        skkuUrls.metadata = metadata
        skkuHtml.metadata = metadata
        skkuMd.metadata = metadata

        # Step 5: Create the table in the database
        logger.debug("Creating tables in the database...")
        Base.metadata.create_all(engine)
        logger.info("Table(s) created successfully in the database.")
    except Exception as e:
        logger.error(f"An error occurred while creating the table: {e}")
        raise

if __name__ == "__main__":
    logger.info("Database module is being run directly, creating tables...")
    create_tables()
    logger.info("Tables created successfully.")
