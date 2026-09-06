import re


AUDIO_TAG = re.compile(r"\s*\[[^\[\]\r\n]+\]\s*")


def strip_audio_tags(text: str) -> str:
    """Remove Eleven v3 performance directions from displayable commentary."""

    return AUDIO_TAG.sub(" ", text).strip()
