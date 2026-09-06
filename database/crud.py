from sqlalchemy.orm import Session

from database.models import Match


def create_match(
    db: Session,
    video_name: str,
    duration: int
):

    match = Match(

        video_name=video_name,

        duration=duration

    )

    db.add(match)

    db.commit()

    db.refresh(match)

    return match