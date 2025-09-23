import uuid
from datetime import datetime

import sqlalchemy.dialects.postgresql as pg
from sqlmodel import Column, Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "user_accounts"

    uid: uuid.UUID = Field(
        sa_column=Column(
            pg.UUID,
            primary_key=True,
            unique=True,
            nullable=False,
            default=uuid.uuid4,
        )
    )

    role: str = Field(
        sa_column=Column(pg.VARCHAR, nullable=False, server_default="user")
    )

    username: str = Field(sa_column=Column(pg.VARCHAR, unique=True, nullable=False))
    first_name: str = Field(sa_column=Column(pg.VARCHAR, nullable=False))
    last_name: str = Field(sa_column=Column(pg.VARCHAR, nullable=False))
    is_verified: bool = Field(default=False)
    email: str = Field(sa_column=Column(pg.VARCHAR, unique=True, nullable=False))
    password_hash: str = Field(sa_column=Column(pg.VARCHAR, nullable=False))
    created_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))
    updated_at: datetime = Field(sa_column=Column(pg.TIMESTAMP, default=datetime.now))

    def __repr__(self) -> str:
        return f"<User {self.username}>"
