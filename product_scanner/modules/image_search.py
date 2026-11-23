"""Product image search module.

Integrates with SearchService (Serper/Brave) to discover product image candidates
and ranks them based on tag style code matches and brand domain signals.
"""

import logging
from typing import List, Optional, Set, Tuple

from config.settings import settings
from modules.models import ImageCandidate, TagFields
from services.search_service import SearchService
from utils.validators import extract_domain, is_valid_url

logger = logging.getLogger(__name__)


class ImageSearcher:
    """Finds product candidate images online using SearchService."""

    @classmethod
    def build_search_queries(cls, tag_fields: TagFields) -> List[str]:
        """Construct ranked query variations from tag metadata."""
        queries: List[str] = []

        brand = tag_fields.brand or ""
        product_type = tag_fields.product_type or ""
        product_name = tag_fields.product_name or ""
        style_code = tag_fields.base_style or tag_fields.style_code or ""
        colour = tag_fields.colour_name or tag_fields.colour_code or ""

        # Variant 1: Brand + Product Type + Style Code
        if brand and product_type and style_code:
            queries.append(f"{brand} {product_type} {style_code}")

        # Variant 2: Brand + Specific Product Name + Style Code
        if brand and product_name and style_code:
            queries.append(f"{brand} {product_name} {style_code}")

        # Variant 3: Brand + Style Code
        if brand and style_code:
            queries.append(f"{brand} {style_code}")

        # Variant 4: Style Code alone
        if style_code and len(style_code) >= 6:
            queries.append(f"{style_code}")

        # Variant 5: Brand + Product Name + Colour
        if brand and product_name and colour:
            queries.append(f"{brand} {product_name} {colour}")

        # Variant 6: Brand + Product Type + Colour
        if brand and product_type and colour:
            queries.append(f"{brand} {product_type} {colour}")

        # Deduplicate preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for q in queries:
            clean_q = " ".join(q.split()).strip()
            if clean_q and clean_q.lower() not in seen:
                seen.add(clean_q.lower())
                deduped.append(clean_q)

        return deduped

    @classmethod
    def search(
        cls, tag_fields: TagFields, app_settings=None, max_candidates: int = 12
    ) -> List[ImageCandidate]:
        """Execute image search using SearchService."""
        queries = cls.build_search_queries(tag_fields)
        if not queries:
            return []

        all_candidates: List[ImageCandidate] = []
        seen_urls: Set[str] = set()

        # Query top 2 constructed search queries
        for q in queries[:2]:
            results = SearchService.search_images(q, num_results=6)
            for res in results:
                url = res.get("image_url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    source_page = res.get("source_page", "")
                    cand = ImageCandidate(
                        url=url,
                        source=f"Search: {res.get('source', 'web')}",
                        title=res.get("title", ""),
                        width=res.get("width"),
                        height=res.get("height"),
                        domain=res.get("domain") or extract_domain(source_page or url),
                        source_page_url=source_page,
                    )
                    all_candidates.append(cand)

        # Style code matching check
        style = (tag_fields.base_style or tag_fields.style_code or "").lower()
        for c in all_candidates:
            haystack = f"{c.title} {c.url} {c.source_page_url}".lower()
            if style and len(style) >= 5 and style in haystack:
                c.style_code_match = True

        all_candidates.sort(key=lambda x: (x.style_code_match, (x.width or 0) * (x.height or 0)), reverse=True)
        return all_candidates[:max_candidates]
