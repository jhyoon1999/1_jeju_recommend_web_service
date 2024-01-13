from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

db_username = "****"
db_password = "****"
db_endpoint = "****"
db_port = 30575
db_name = "****"

db_url = f"mariadb+pymysql://{db_username}:{db_password}@{db_endpoint}:{db_port}/{db_name}?charset=utf8"
engine = create_engine(db_url, echo=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()