from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import DateTime
from sqlalchemy.sql import func

from database.database import Base


class Match(Base):

    __tablename__ = "matches"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    video_name = Column(
        String,
        nullable=False
    )

    status = Column(
        String,
        default="uploaded"
    )

    duration = Column(
        Integer
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )