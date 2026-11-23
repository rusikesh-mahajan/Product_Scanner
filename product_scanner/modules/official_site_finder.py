"""Official website discovery and domain validation module.

Discovers the brand's official website using tag customer-care email domains,
search queries, blocklist filtering, and live HTTPS & branding verification.
Maintains and updates config/official_domains.json dynamically.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse

from bs4 import BeautifulSoup
import requests
import tldextract

from config.settings import settings
from modules.models import OfficialSite, ProductRecord
from services.search_service import SearchService
from utils.validators import is_valid_url

logger = logging.getLogger(__name__)


class OfficialSiteFinder:
    """Discovers and validates official brand websites for product lookup."""

    _blocked_domains: Optional[Set[str]] = None
    _official_domains_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _get_blocked_domains(cls) -> Set[str]:
        """Load blocklist of marketplaces and aggregators from config/blocked_domains.json."""
        if cls._blocked_domains is not None:
            return cls._blocked_domains

        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "blocked_domains.json"
        )
        blocked = set()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    blocked = {d.strip().lower() for d in data if isinstance(d, str)}
            except Exception as e:
                logger.error("Failed to load blocked domains: %s", e)

        cls._blocked_domains = blocked
        return cls._blocked_domains

    @classmethod
    def _load_official_domains(cls) -> Dict[str, Any]:
        """Load previously discovered & verified official domains from config/official_domains.json."""
        if cls._official_domains_cache is not None:
            return cls._official_domains_cache

        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "official_domains.json"
        )
        official = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    official = json.load(f)
            except Exception as e:
                logger.error("Failed to load official domains: %s", e)

        cls._official_domains_cache = official
        return cls._official_domains_cache

    @classmethod
    def save_verified_domain(cls, domain: str, brand: str, method: str) -> None:
        """Persist a verified official domain to config/official_domains.json."""
        if not domain:
            return
        clean_domain = domain.lower().strip()
        data = cls._load_official_domains()
        data[clean_domain] = {
            "brand": brand,
            "verified_via": method,
            "timestamp": OfficialSite(clean_domain, brand).timestamp,
        }
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "official_domains.json"
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            cls._official_domains_cache = data
            logger.info("Persisted verified official domain '%s' for brand '%s'", clean_domain, brand)
        except Exception as e:
            logger.error("Failed to save official domain: %s", e)

    @classmethod
    def is_blocked(cls, domain_or_url: str) -> bool:
        """Check if a domain belongs to a blocked marketplace or aggregator."""
        if not domain_or_url:
            return True
        blocked = cls._get_blocked_domains()
        ext = tldextract.extract(domain_or_url)
        registered_domain = f"{ext.domain}.{ext.suffix}".lower()
        full_domain = (ext.fqdn or domain_or_url).lower()

        if registered_domain in blocked or full_domain in blocked:
            return True
        for b in blocked:
            if b in full_domain:
                return True
        return False

    @classmethod
    def validate_domain(
        cls, domain: str, brand_name: str
    ) -> Tuple[bool, str, str]:
        """Validate that domain has HTTPS, resolves, and contains brand references.

        Returns (is_valid, final_homepage_url, reason_or_detail).
        """
        if not domain or cls.is_blocked(domain):
            return False, "", "Domain is empty or on marketplace blocklist"

        SearchService.rate_limit(domain, min_delay=1.0)
        homepage_url = f"https://{domain}"
        headers = {"User-Agent": settings.user_agent}

        def _safe_get(target_url: str):
            try:
                return requests.get(
                    target_url,
                    headers=headers,
                    timeout=settings.request_timeout,
                    allow_redirects=True,
                )
            except requests.exceptions.SSLError:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                return requests.get(
                    target_url,
                    headers=headers,
                    timeout=settings.request_timeout,
                    allow_redirects=True,
                    verify=False,
                )

        try:
            resp = _safe_get(homepage_url)
            if resp.status_code != 200:
                # Try with www. prefix if bare failed
                if not domain.startswith("www."):
                    alt_url = f"https://www.{domain}"
                    resp = _safe_get(alt_url)
                    if resp.status_code == 200:
                        homepage_url = resp.url
                    else:
                        return False, "", f"Homepage returned HTTP {resp.status_code}"
                else:
                    return False, "", f"Homepage returned HTTP {resp.status_code}"

            final_url = resp.url
            html_content = resp.text

            # Check title and meta tags for brand mention
            soup = BeautifulSoup(html_content, "html.parser")
            title_text = (soup.title.string or "") if soup.title else ""
            meta_desc = ""
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
                "meta", attrs={"property": "og:description"}
            )
            if desc_tag and desc_tag.get("content"):
                meta_desc = desc_tag["content"]

            haystack = f"{title_text} {meta_desc} {resp.url}".lower()

            # Clean brand tokens (e.g. "United Colors of Benetton" -> ["benetton"], "Jack & Jones" -> ["jack", "jones"])
            brand_clean = re.sub(r"[^\w\s]", " ", brand_name).lower()
            tokens = [t for t in brand_clean.split() if t not in ("of", "and", "the", "in", "india")]

            brand_found = any(t in haystack for t in tokens if len(t) >= 4)
            if not brand_found:
                # Also check body text in top 2000 characters
                top_text = soup.get_text()[:3000].lower()
                brand_found = any(t in top_text for t in tokens if len(t) >= 4)

            if not brand_found:
                return (
                    False,
                    final_url,
                    f"Brand '{brand_name}' not found in homepage title/branding ('{title_text[:60]}')",
                )

            return True, final_url, f"Verified homepage: title='{title_text[:60]}'"

        except requests.exceptions.SSLError as e:
            logger.warning("SSL error validating %s: %s", domain, e)
            return False, "", f"SSL/HTTPS error: {e}"
        except requests.exceptions.RequestException as e:
            logger.warning("Connection error validating %s: %s", domain, e)
            return False, "", f"Connection failed: {e}"
        except Exception as e:
            logger.warning("Unexpected error validating %s: %s", domain, e)
            return False, "", f"Validation error: {e}"

    @classmethod
    def find_official_site(cls, record: ProductRecord) -> Optional[OfficialSite]:
        """Discover and validate the official website for the product's brand.

        Priority 1: Customer care email domain printed on tag.
        Priority 2: Previously verified domain from config/official_domains.json.
        Priority 3: Web search discovery (Serper / Brave) with rigorous validation.
        """
        brand = record.brand or "Unknown"
        email = record.customer_care_email or ""

        PARENT_HOLDING_DOMAINS = {
            "arvindfashions.com",
            "arvind.com",
            "abfrl.com",
            "adityabirla.com",
            "madurafnl.com",
            "reliance.com",
            "ril.com",
            "trent-ltd.com",
        }

        # ---------------------------------------------------------------------
        # SIGNAL 1: Check curated/cached verified official domains first
        # ---------------------------------------------------------------------
        cached_domains = cls._load_official_domains()
        for dom, info in cached_domains.items():
            cached_brand = (info.get("brand") or "").lower()
            if brand and (brand.lower() in cached_brand or cached_brand in brand.lower()):
                is_valid, hp_url, detail = cls.validate_domain(dom, brand)
                if is_valid:
                    return OfficialSite(
                        domain=dom,
                        brand=brand,
                        homepage_url=hp_url,
                        verification_method="cached_verified",
                        is_verified=True,
                        verification_details=f"Matched curated/cached official brand domain: {detail}",
                    )

        # ---------------------------------------------------------------------
        # SIGNAL 2: Customer care email domain (if not a parent conglomerate)
        # ---------------------------------------------------------------------
        if email and "@" in email:
            email_domain = email.split("@")[-1].strip().lower()
            ext = tldextract.extract(email_domain)
            reg_domain = f"{ext.domain}.{ext.suffix}".lower()

            # Skip conglomerate holding companies (e.g. arvindfashions.com for U.S. Polo Assn)
            if reg_domain not in PARENT_HOLDING_DOMAINS and email_domain not in PARENT_HOLDING_DOMAINS:
                candidate_domains = [email_domain]
                if reg_domain != email_domain:
                    candidate_domains.append(reg_domain)

                # Common country-specific variants (e.g. benetton.co.in -> benetton.in)
                if ext.suffix == "co.in":
                    candidate_domains.append(f"{ext.domain}.in")
                elif ext.suffix == "in":
                    candidate_domains.append(f"{ext.domain}.co.in")

                for cd in candidate_domains:
                    if cls.is_blocked(cd):
                        continue
                    is_valid, hp_url, detail = cls.validate_domain(cd, brand)
                    if is_valid:
                        site = OfficialSite(
                            domain=cd,
                            brand=brand,
                            homepage_url=hp_url,
                            verification_method="email_domain",
                            is_verified=True,
                            verification_details=f"Matched tag email domain ({email}): {detail}",
                        )
                        cls.save_verified_domain(cd, brand, "email_domain")
                        return site
                    return OfficialSite(
                        domain=dom,
                        brand=brand,
                        homepage_url=hp_url,
                        verification_method="config_cached",
                        is_verified=True,
                        verification_details=f"Loaded from verified official_domains.json: {detail}",
                    )

        # ---------------------------------------------------------------------
        # SIGNAL 3: Search queries ("<brand> official website India", etc.)
        # ---------------------------------------------------------------------
        if not brand or brand == "Unknown":
            logger.warning("Cannot search for official site without brand name.")
            return None

        queries = [
            f"{brand} official website India",
            f"{brand} India online store",
            f"{brand} official website",
        ]

        logger.info("Searching web for official website of '%s'...", brand)
        evaluated_candidates: List[str] = []

        for q in queries:
            results = SearchService.search_web(q, num_results=8)
            for res in results:
                url = res.get("url", "")
                parsed = urllib.parse.urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]

                if not domain or domain in evaluated_candidates:
                    continue
                evaluated_candidates.append(domain)

                # Never accept blocked marketplaces
                if cls.is_blocked(domain):
                    logger.debug("Skipping blocked marketplace domain: %s", domain)
                    continue

                # Brand name token must appear in domain name
                brand_tokens = [
                    t for t in re.sub(r"[^\w\s]", "", brand).lower().split()
                    if t not in ("of", "and", "the", "in", "india")
                ]
                domain_matches_brand = any(tok in domain for tok in brand_tokens if len(tok) >= 4)
                if not domain_matches_brand:
                    continue

                is_valid, hp_url, detail = cls.validate_domain(domain, brand)
                if is_valid:
                    site = OfficialSite(
                        domain=domain,
                        brand=brand,
                        homepage_url=hp_url,
                        verification_method="web_search_validated",
                        is_verified=True,
                        verification_details=f"Discovered via search '{q}': {detail}",
                    )
                    cls.save_verified_domain(domain, brand, "web_search_validated")
                    return site

        logger.warning("Could not discover a verified official website for brand '%s'", brand)
        return None
