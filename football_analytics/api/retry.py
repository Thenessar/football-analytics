import random
import time
from email.utils import parsedate_to_datetime
from typing import Optional

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (retry_at - retry_at.now(retry_at.tzinfo)).total_seconds())


def backoff_delay(
    attempt: int,
    *,
    base_seconds: float = 0.5,
    max_seconds: float = 30.0,
    jitter_seconds: float = 0.25,
    retry_after: Optional[str] = None,
) -> float:
    retry_after_delay = parse_retry_after(retry_after)
    if retry_after_delay is not None:
        return min(retry_after_delay, max_seconds)
    exponential = min(max_seconds, base_seconds * (2 ** max(0, attempt - 1)))
    return exponential + random.uniform(0, jitter_seconds)


def sleep_before_retry(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)

