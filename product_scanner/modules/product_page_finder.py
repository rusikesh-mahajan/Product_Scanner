"""Product page discovery module.

Searches the verified official brand website for the specific product page
using targeted `site:<domain>` queries and ranking candidates by style/SKU match.
"""

import logging
from typing import List, Optional, Tuple
import urllib.parse

from modules.models import OfficialSite, ProductPage, ProductRecord
from services.search_service import SearchService
from utils.validators import extract_domain, is_valid_url

logger = logging.getLogger(__name__)


class ProductPageFinder:
    """Finds exact and candidate product pages on a verified brand domain."""

    @classmethod
    def find_product_pages(
        cls, record: ProductRecord, official_site: OfficialSite
    ) -> List[ProductPage]:
        """Search the official domain for matching product pages.

        Order of search queries:
        1. site:<domain> "<base_style>"
        2. site:<domain> "<style_code>"
        3. site:<domain> "<style_variant>"
        4. site:<domain> "<product_name>" "<colour>"
        5. site:<domain> "<product_type>" "<colour>"
        """
        domain = official_site.domain
        candidates: List[ProductPage] = []
        seen_urls = set()

        style = record.base_style or record.style_code or ""
        pname = record.product_name or ""
        ptype = record.product_type or ""
        colour = record.colour_name or record.colour_code or ""

        queries_to_run: List[Tuple[str, str, float]] = []

        # Exact style query
        if style:
            queries_to_run.append((f'site:{domain} "{style}"', "exact_style", 1.0))
            queries_to_run.append((f"site:{domain} {style}", "loose_style", 0.9))

        # Variants for OCR confusion (e.g. 0 vs O, 1 vs I)
        for variant in record.style_variants[:3]:
            if variant and variant != style:
                queries_to_run.append(
                    (f'site:{domain} "{variant}"', "style_variant", 0.85)
                )

        # Product name + colour query
        if pname and colour:
            queries_to_run.append(
                (f'site:{domain} "{pname}" {colour}', "name_colour", 0.7)
            )
        elif pname:
            queries_to_run.append((f'site:{domain} "{pname}"', "name_only", 0.6))

        # Product type + colour query
        if ptype and colour:
            queries_to_run.append(
                (f'site:{domain} "{ptype}" {colour}', "type_colour", 0.5)
            )

        for query_str, q_type, base_confidence in queries_to_run:
            logger.info("Running product page search: %s", query_str)
            results = SearchService.search_web(query_str, num_results=5)

            for res in results:
                url = res.get("url", "")
                if not url or not is_valid_url(url):
                    continue

                res_domain = extract_domain(url).lower()
                # Ensure result is on official domain or subdomains
                if domain not in res_domain:
                    continue

                # Block non-product or internal/automation subdomains
                blocked_subdomains = (
                    "automation",
                    "intranet",
                    "portal",
                    "vpn",
                    "internal",
                    "erp",
                    "dev.",
                    "staging.",
                    "api.",
                    "mail.",
                )
                if any(sub in res_domain for sub in blocked_subdomains):
                    logger.debug("Skipping internal subdomain URL: %s", url)
                    continue

                # Ignore non-product URLs like cart, checkout, account, search, terms, download portals
                path_lower = urllib.parse.urlparse(url).path.lower()
                if any(
                    kw in path_lower
                    for kw in (
                        "/cart",
                        "/checkout",
                        "/account",
                        "/login",
                        "/policy",
                        "/terms",
                        "/contact",
                        "/about",
                        "/faq",
                        "/downloadui",
                        "/download",
                        "/portal",
                        "/admin",
                        "/careers",
                        "/investor",
                        "/press",
                        "/news",
                        "/corporate",
                    )
                ):
                    continue

                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = res.get("title", "")
                snippet = res.get("snippet", "")
                haystack = f"{url} {title} {snippet}".lower()

                # Check if style code is present in URL or title
                style_match = False
                if style and len(style) >= 4:
                    style_match = (style.lower() in haystack)
                if not style_match:
                    for v in record.style_variants:
                        if v and len(v) >= 4 and v.lower() in haystack:
                            style_match = True
                            break

                # Check if product name or type is present
                name_match = False
                if pname and len(pname) >= 4:
                    name_match = pname.lower() in haystack
                elif ptype and len(ptype) >= 4:
                    name_match = ptype.lower() in haystack

                confidence = base_confidence
                if style_match:
                    confidence = max(confidence, 0.95)
                elif name_match:
                    confidence = max(confidence, 0.70)

                page = ProductPage(
                    url=url,
                    domain=res_domain,
                    title=title,
                    snippet=snippet,
                    style_match=style_match,
                    name_match=name_match,
                    confidence=confidence,
                    search_query=query_str,
                )
                candidates.append(page)

            # If we found exact style matches, we can stop querying
            if any(c.style_match for c in candidates):
                logger.info("Found exact style match page(s) on %s", domain)
                break

        # Sort by confidence descending
        candidates.sort(key=lambda x: (x.style_match, x.confidence), reverse=True)
        return candidates
