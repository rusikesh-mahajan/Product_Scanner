"""General-purpose helper utilities."""

import functools
import logging
import os
import re
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable:
    """Decorator that retries a function with exponential back-off."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "Attempt %d/%d for %s failed: %s — retrying in %.1fs",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts for %s failed: %s",
                            max_retries,
                            func.__name__,
                            exc,
                        )
            if last_exc is not None:
                raise last_exc
            return None  # unreachable but keeps mypy happy

        return wrapper

    return decorator


def format_price(amount: Optional[float], currency: str = "INR") -> str:
    """Return a human-friendly price string like ``₹3,299.00``."""
    if amount is None:
        return ""
    symbols = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, f"{currency} ")
    return f"{symbol}{amount:,.2f}"


def sanitize_filename(name: str) -> str:
    """Turn an arbitrary string into a safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name[:200] if name else "unnamed"


def ensure_dir(path: str) -> str:
    """Create *path* if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def truncate_text(text: str, max_length: int = 100) -> str:
    """Shorten *text* to *max_length* with an ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."

