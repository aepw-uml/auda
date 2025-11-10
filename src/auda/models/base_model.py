from sqlalchemy import Enum
from sqlalchemy.orm import declarative_base

# SQLAlchemy base; all models should inherit from this
SABase = declarative_base()

# Enum type for SQLAlchemy
SqlalchemyEnum = Enum
