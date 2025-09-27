import sqlalchemy.dialects.postgresql as pg
from sqlalchemy.sql import func
from sqlmodel import SQLModel, Field, Column
import uuid
from datetime import datetime


class Prediction(SQLModel, table=True):
    __tablename__ = "predictions"

    uid: uuid.UUID = Field(
        sa_column=Column(
            pg.UUID, primary_key=True, unique=True, nullable=False, default=uuid.uuid4
        )
    )

    image_path: str = Field(sa_column=Column(pg.TEXT, nullable=False, unique=True))

    predicted_class: int = Field(sa_column=Column(pg.INTEGER, nullable=False))
    predicted_class_name: str = Field(sa_column=Column(pg.TEXT, nullable=False))
    confidence: float = Field(sa_column=Column(pg.FLOAT, nullable=False))

    # foreign_key="user_accounts.uid" later when user management is implemented
    user_uid: uuid.UUID = Field(default=None, nullable=False)

    created_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))
