"""Search service integrating Serper.dev and Brave Search APIs.

Provides web search, image search, robots.txt checking, per-domain rate limiting,
and in-memory request caching.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.robotparser

import requests

from config.settings import settings
from utils.validators import extract_domain, is_valid_url

logger = logging.getLogger(__name__)


class SearchService:
    """Manages search queries across Serper and Brave with rate limiting and compliance."""

    _last_domain_request_time: Dict[str, float] = {}
    _robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
    _search_cache: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def rate_limit(cls, domain: str, min_delay: Optional[float] = None) -> None:
        """Enforce domain-specific delay between consecutive requests."""
        if not domain:
            return
        delay = min_delay if min_delay is not None else settings.rate_limit_delay_seconds
        last_time = cls._last_domain_request_time.get(domain, 0.0)
        elapsed = time.time() - last_time
        if elapsed < delay:
            sleep_needed = delay - elapsed
            logger.debug("Rate limiting %s: sleeping for %.2fs", domain, sleep_needed)
            time.sleep(sleep_needed)
        cls._last_domain_request_time[domain] = time.time()

    @classmethod
    def can_fetch(cls, url: str, user_agent: Optional[str] = None) -> bool:
        """Check robots.txt compliance before fetching a URL."""
        if not is_valid_url(url):
            return False
        ua = user_agent or settings.user_agent
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        base_url = f"{parsed.scheme}://{domain}"

        if domain not in cls._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{base_url}/robots.txt"
            try:
                cls.rate_limit(domain, min_delay=0.5)
                rp.set_url(robots_url)
                rp.read()
                cls._robots_cache[domain] = rp
            except Exception as e:
                logger.debug("Could not read robots.txt for %s: %s (defaulting to allow)", domain, e)
                # If robots.txt cannot be fetched or parsed, default to permissive
                rp.allow_all = True
                cls._robots_cache[domain] = rp

        rp = cls._robots_cache.get(domain)
        if rp and not rp.can_fetch(ua, url):
            logger.warning("robots.txt disallows fetching: %s for %s", url, ua)
            return False
        return True

    @classmethod
    def search_web(
        cls, query: str, num_results: int = 10, domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a web search query via Serper or Brave.

        Returns list of dicts: [{"title": ..., "url": ..., "snippet": ..., "domain": ...}]
        """
        cache_key = f"web:{query}:{num_results}"
        if cache_key in cls._search_cache:
            logger.debug("Returning cached web search for: %s", query)
            return cls._search_cache[cache_key]

        results: List[Dict[str, Any]] = []

        # 1. Try Serper Web Search (if key provided)
        if settings.has_serper_key:
            results = cls._search_serper_web(query, num_results)

        # 2. Try Brave Web Search (if no results or key not set)
        if not results and settings.has_brave_key:
            results = cls._search_brave_web(query, num_results)

        # 3. Try Tavily Web Search (if no results or key not set)
        if not results and settings.has_tavily_key:
            results = cls._search_tavily_web(query, num_results, domain_filter=domain_filter)

        # Post-filter if domain_filter is requested
        if domain_filter:
            results = [r for r in results if domain_filter.lower() in (r.get("domain") or "").lower()]

        cls._search_cache[cache_key] = results
        return results

    @classmethod
    def _search_serper_web(cls, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """Query Serper.dev Google Web Search endpoint."""
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num}
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=settings.request_timeout
            )
            if resp.status_code != 200:
                logger.warning("Serper web search returned HTTP %d: %s", resp.status_code, resp.text[:150])
                return []
            data = resp.json()
            items = data.get("organic", [])
            out = []
            for item in items:
                link = item.get("link", "")
                if link and is_valid_url(link):
                    out.append({
                        "title": item.get("title", ""),
                        "url": link,
                        "snippet": item.get("snippet", ""),
                        "domain": extract_domain(link),
                        "source": "serper_web",
                    })
            return out
        except Exception as e:
            logger.error("Serper web search error: %s", e)
            return []

    @classmethod
    def _search_brave_web(cls, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """Query Brave Web Search API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": settings.brave_api_key,
        }
        params = {"q": query, "count": count}
        try:
            resp = requests.get(
                url, headers=headers, params=params, timeout=settings.request_timeout
            )
            if resp.status_code != 200:
                logger.warning("Brave web search returned HTTP %d: %s", resp.status_code, resp.text[:150])
                return []
            data = resp.json()
            web_results = data.get("web", {}).get("results", [])
            out = []
            for item in web_results:
                link = item.get("url", "")
                if link and is_valid_url(link):
                    out.append({
                        "title": item.get("title", ""),
                        "url": link,
                        "snippet": item.get("description", ""),
                        "domain": extract_domain(link),
                        "source": "brave_web",
                    })
            return out
        except Exception as e:
            logger.error("Brave web search error: %s", e)
            return []

    @classmethod
    def search_images(cls, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Execute image search query via Serper or Brave."""
        cache_key = f"img:{query}:{num_results}"
        if cache_key in cls._search_cache:
            return cls._search_cache[cache_key]

        results: List[Dict[str, Any]] = []
        if settings.has_serper_key:
            results = cls._search_serper_images(query, num_results)
        if not results and settings.has_brave_key:
            results = cls._search_brave_images(query, num_results)
        if not results and settings.has_tavily_key:
            results = cls._search_tavily_images(query, num_results)

        cls._search_cache[cache_key] = results
        return results

    @classmethod
    def _search_serper_images(cls, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """Query Serper.dev Google Images endpoint."""
        url = "https://google.serper.dev/images"
        headers = {
            "X-API-KEY": settings.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=settings.request_timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            out = []
            for item in data.get("images", []):
                img_url = item.get("imageUrl")
                if img_url and is_valid_url(img_url):
                    out.append({
                        "image_url": img_url,
                        "source_page": item.get("link", ""),
                        "title": item.get("title", ""),
                        "width": item.get("imageWidth"),
                        "height": item.get("imageHeight"),
                        "domain": extract_domain(item.get("link") or img_url),
                        "source": "serper_images",
                    })
            return out
        except Exception as e:
            logger.error("Serper images error: %s", e)
            return []

    @classmethod
    def _search_brave_images(cls, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """Query Brave Images Search endpoint."""
        url = "https://api.search.brave.com/res/v1/images/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": settings.brave_api_key,
        }
        params = {"q": query, "count": count}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=settings.request_timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            out = []
            for item in data.get("results", []):
                props = item.get("properties", {})
                img_url = props.get("url") or item.get("thumbnail", {}).get("src")
                if img_url and is_valid_url(img_url):
                    source_page = item.get("url", "")
                    out.append({
                        "image_url": img_url,
                        "source_page": source_page,
                        "title": item.get("title", ""),
                        "width": props.get("width"),
                        "height": props.get("height"),
                        "domain": extract_domain(source_page or img_url),
                        "source": "brave_images",
                    })
            return out
        except Exception as e:
            logger.error("Brave images error: %s", e)
            return []

    @classmethod
    def _search_tavily_web(
        cls, query: str, num: int = 10, domain_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query Tavily Search API endpoint."""
        url = "https://api.tavily.com/search"
        payload: Dict[str, Any] = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_images": False,
            "max_results": min(num, 10),
        }
        if domain_filter:
            payload["include_domains"] = [domain_filter]

        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=settings.request_timeout
            )
            if resp.status_code != 200:
                logger.warning("Tavily search returned HTTP %d: %s", resp.status_code, resp.text[:150])
                return []
            data = resp.json()
            out = []
            for item in data.get("results", []):
                link = item.get("url", "")
                if link and is_valid_url(link):
                    out.append({
                        "title": item.get("title", ""),
                        "url": link,
                        "snippet": item.get("content", ""),
                        "domain": extract_domain(link),
                        "source": "tavily_web",
                    })
            return out
        except Exception as e:
            logger.error("Tavily search error: %s", e)
            return []

    @classmethod
    def _search_tavily_images(cls, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """Query Tavily Image Search via /search endpoint with include_images=True."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": settings.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "include_images": True,
            "max_results": min(num, 10),
        }
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=settings.request_timeout
            )
            if resp.status_code != 200:
                logger.warning("Tavily image search returned HTTP %d: %s", resp.status_code, resp.text[:150])
                return []
            data = resp.json()
            out = []
            raw_images = data.get("images", [])
            for item in raw_images:
                img_url = item if isinstance(item, str) else item.get("url", "")
                desc = item.get("description", "") if isinstance(item, dict) else ""
                if img_url and is_valid_url(img_url):
                    out.append({
                        "image_url": img_url,
                        "source_page": "",
                        "title": desc or "Tavily product image",
                        "width": None,
                        "height": None,
                        "domain": extract_domain(img_url),
                        "source": "tavily_images",
                    })
            return out
        except Exception as e:
            logger.error("Tavily image search error: %s", e)
            return []
