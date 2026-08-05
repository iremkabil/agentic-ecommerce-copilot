"""Deterministic shipping calculator (rules table).

No LLM, no randomness: identical inputs always produce identical cost and ETA.
This is exactly the kind of logic you must NOT delegate to the model — it's a
pure function the agent *calls*. ``RATES`` below is the single source of truth
and mirrors the numbers written in ``data/policies.md``.

Two layers, mirroring the retrieval design:
* ``compute_shipping`` — a pure function (no DB), trivially testable.
* ``shipping_calculator`` — the tool entrypoint that looks up item weights and
  prices from the DB, then delegates to ``compute_shipping``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sqlalchemy.orm import Session

from copilot.db.models import Product

# Could move to settings; kept here so the whole rules table reads in one place.
HOME_COUNTRY = "US"
FREE_SHIPPING_THRESHOLD = 40.0
CURRENCY = "USD"

_EU = {"DE", "FR", "ES", "IT", "NL", "IE", "BE", "AT", "PT", "FI", "SE", "DK", "PL", "GR", "CZ"}
SUPPORTED_COUNTRIES = {"US", "GB"} | _EU


@dataclass(frozen=True)
class Rate:
    base: float
    surcharge_per_500g_over_500g: float
    min_days: int
    max_days: int


# (zone, method) -> Rate
RATES: dict[tuple[str, str], Rate] = {
    ("domestic", "standard"): Rate(4.90, 0.0, 3, 5),
    ("domestic", "express"): Rate(9.90, 0.0, 1, 2),
    ("international", "standard"): Rate(14.90, 3.0, 7, 14),
    ("international", "express"): Rate(24.90, 3.0, 4, 7),
}


def resolve_zone(country: str) -> str | None:
    """Map a country code to a shipping zone, or None if we don't ship there."""
    c = (country or "").strip().upper()
    if c not in SUPPORTED_COUNTRIES:
        return None
    return "domestic" if c == HOME_COUNTRY else "international"


def compute_shipping(
    *,
    country: str,
    total_weight_g: int = 0,
    subtotal: float = 0.0,
    method: str = "standard",
) -> dict:
    """Pure shipping computation. Returns a JSON-ready dict (or an error dict)."""
    method = (method or "standard").lower()
    zone = resolve_zone(country)
    if zone is None:
        return {
            "error": "unsupported_destination",
            "supported_countries": sorted(SUPPORTED_COUNTRIES),
        }
    if (zone, method) not in RATES:
        return {"error": "unsupported_method", "supported_methods": ["standard", "express"]}

    rate = RATES[(zone, method)]
    over = max(0, total_weight_g - 500)
    surcharge = math.ceil(over / 500) * rate.surcharge_per_500g_over_500g
    cost = rate.base + surcharge

    free_applied = False
    if zone == "domestic" and method == "standard" and subtotal >= FREE_SHIPPING_THRESHOLD:
        cost = 0.0
        free_applied = True

    return {
        "method": method,
        "zone": zone,
        "cost": round(cost, 2),
        "currency": CURRENCY,
        "free_shipping_applied": free_applied,
        "estimated_delivery_business_days": {"min": rate.min_days, "max": rate.max_days},
        "dispatch_business_days": {"min": 1, "max": 2},
    }


def shipping_calculator(
    session: Session,
    *,
    destination_country: str,
    items: list[dict],
    method: str = "standard",
) -> dict:
    """Tool entrypoint. ``items`` = [{"product_id": str, "quantity": int}, ...].

    Looks up each item's weight and price from the DB to derive total weight and
    subtotal, then computes shipping. Unknown ids are reported, not fatal.
    """
    total_weight = 0
    subtotal = 0.0
    unknown: list[str] = []

    for item in items:
        product = session.get(Product, item["product_id"])
        quantity = int(item.get("quantity", 1))
        if product is None:
            unknown.append(item["product_id"])
            continue
        total_weight += (product.weight_grams or 0) * quantity
        subtotal += product.price * quantity

    result = compute_shipping(
        country=destination_country,
        total_weight_g=total_weight,
        subtotal=subtotal,
        method=method,
    )
    result["subtotal"] = round(subtotal, 2)
    result["total_weight_g"] = total_weight
    if unknown:
        result["unknown_product_ids"] = unknown
    return result
