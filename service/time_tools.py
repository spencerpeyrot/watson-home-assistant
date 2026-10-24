"""Read-only current-time lookup without a model or network request."""

from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field


PLACE_ZONES = {
    "nyc": "America/New_York", "new york": "America/New_York",
    "new york city": "America/New_York", "london": "Europe/London",
    "paris": "Europe/Paris", "tokyo": "Asia/Tokyo",
    "chicago": "America/Chicago", "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "boston": "America/New_York", "miami": "America/New_York",
    "berlin": "Europe/Berlin", "rome": "Europe/Rome",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
    "dubai": "Asia/Dubai", "toronto": "America/Toronto",
    "vancouver": "America/Vancouver", "utc": "UTC", "gmt": "Etc/GMT",
}


class CurrentTimeArgs(BaseModel):
    place: str = Field(min_length=1, max_length=100)


def get_current_time(place: str) -> dict[str, str]:
    """Resolve explicit city aliases or IANA zones; never guess unknown places."""

    cleaned = place.strip().rstrip(".?!").strip()
    zone_name = PLACE_ZONES.get(cleaned.casefold())
    if zone_name is None and "/" in cleaned:
        zone_name = cleaned
    if zone_name is None:
        return {"status": "needs_clarification", "answer": "Which city or IANA timezone do you mean?"}
    try:
        zone = ZoneInfo(zone_name)
    except (ValueError, ZoneInfoNotFoundError):
        return {"status": "needs_clarification", "answer": "I couldn't resolve that timezone. Which city do you mean?"}
    now = datetime.now(timezone.utc).astimezone(zone)
    clock = f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    return {
        "status": "succeeded", "timezone": zone_name,
        "local_datetime": now.isoformat(timespec="seconds"),
        "answer": f"It's {clock} in {cleaned}, {now.strftime('%A, %B')} {now.day} ({now.tzname()}).",
    }


def fast_time_answer(request_text: str) -> str | None:
    """Only handle a complete, single current-time question."""

    place = current_time_place(request_text)
    if place is None:
        return None
    return get_current_time(place)["answer"]


def current_time_place(request_text: str) -> str | None:
    """Parse a complete clock question without sampling the clock yet."""

    match = re.fullmatch(
        r"\s*(?:hey\s+watson[,.:]?\s*)?(?:what(?:['’]s| is) (?:the |current )?time|what time is it) in ([\w /-]{1,100})[?.!]?\s*",
        request_text, re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).strip()
