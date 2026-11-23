"""Abstract base class for barcode product lookup API providers."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

from config.settings import settings
from modules.models import ProductResult

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Base class for all product lookup API providers."""

    name: str = "BaseProvider"

    @abstractmethod
    def lookup(self, barcode: str) -> Optional[ProductResult]:
        """Look up product information by barcode.

        Returns ProductResult if found, or None if not found or on error.
        Must never raise an unhandled exception to caller.
        """
        pass

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Optional[requests.Response]:
        """Execute HTTP request with defensive error handling."""
        timeout_val = timeout or settings.request_timeout
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout_val,
            )

            if resp.status_code == 200:
                return resp

            if resp.status_code == 404:
                logger.info("%s: barcode not found (404)", self.name)
                return None

            if resp.status_code in (401, 403):
                logger.warning(
                    "%s: authentication error (HTTP %d). Check your API key.",
                    self.name,
                    resp.status_code,
                )
                return None

            if resp.status_code == 429:
                logger.warning(
                    "%s: rate limit reached (HTTP 429).", self.name
                )
                return None

            logger.warning(
                "%s: received non-success status code %d from %s",
                self.name,
                resp.status_code,
                url,
            )
            return None

        except requests.Timeout:
            logger.warning("%s: request timed out after %ds", self.name, timeout_val)
            return None
        except requests.RequestException as e:
            logger.error("%s: request network error: %s", self.name, e)
            return None
        except Exception as e:
            logger.error("%s: unexpected error during request: %s", self.name, e)
            return None

