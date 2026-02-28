"""CSV → typed Product objects for query generation and ground truth."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StockVariant:
    """A single size/color inventory entry."""
    size: str
    color: str
    in_stock: bool


@dataclass
class Product:
    """Typed representation of a single product row from ground truth CSV."""

    # Identification
    product_id: str
    product_name: str

    # Core facts
    price: float
    currency: str
    material: str
    care_instructions: str
    return_policy: str
    shipping_standard_days: int
    shipping_express_days: int

    # Optional core facts
    hardware_color: str = ""
    fit_type: str = ""
    dimensions_cm: str = ""
    laptop_compatibility: str = ""
    sizing_notes: str = ""
    model_height: str = ""
    model_weight: str = ""
    model_size_worn: str = ""

    # Stock matrix (parsed from stock_{SIZE}_{COLOR} columns)
    stock: list[StockVariant] = field(default_factory=list)

    # Intelligence layer
    occasion_tags: list[str] = field(default_factory=list)
    warmth_score: float = 0.0
    waterproof_score: float = 0.0
    office_appropriate_score: float = 0.0
    weight_grams: float = 0.0
    is_lightest_in_catalog: bool = False
    is_most_packable: bool = False
    catalog_rank_by_weight: int = 0

    # Adversarial
    non_existent_colors: list[str] = field(default_factory=list)

    # Derived helpers
    @property
    def available_colors(self) -> list[str]:
        return sorted({v.color for v in self.stock if v.in_stock})

    @property
    def available_sizes(self) -> list[str]:
        return sorted({v.size for v in self.stock if v.in_stock})

    @property
    def oos_variants(self) -> list[StockVariant]:
        return [v for v in self.stock if not v.in_stock]

    @property
    def in_stock_variants(self) -> list[StockVariant]:
        return [v for v in self.stock if v.in_stock]

    def is_in_stock(self, size: str, color: str) -> bool | None:
        """Return True/False if variant exists, None if variant not applicable."""
        for v in self.stock:
            if v.size.lower() == size.lower() and v.color.lower() == color.lower():
                return v.in_stock
        return None


def _parse_stock_columns(row: dict[str, str]) -> list[StockVariant]:
    """Extract stock_{SIZE}_{COLOR} columns into StockVariant list."""
    variants: list[StockVariant] = []
    for key, value in row.items():
        if not key.startswith("stock_") or not value.strip():
            continue
        parts = key[len("stock_"):].split("_", 1)
        if len(parts) != 2:
            continue
        size, color = parts
        variants.append(StockVariant(
            size=size,
            color=color,
            in_stock=value.strip().lower() == "true",
        ))
    return variants


def _safe_float(val: str | None, default: float = 0.0) -> float:
    if not val or not val.strip():
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _safe_int(val: str | None, default: int = 0) -> int:
    if not val or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _safe_str(val: str | None, default: str = "") -> str:
    return val.strip() if val and val.strip() else default


def _safe_bool(val: str | None) -> bool:
    return val.strip().lower() == "true" if val and val.strip() else False


def _split_csv(val: str | None) -> list[str]:
    if not val or not val.strip():
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


def load_products(csv_path: str | Path) -> list[Product]:
    """Load and validate a ground truth CSV, returning typed Product objects.

    Raises ValueError with a clear message if required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        # Skip comment rows (lines starting with #)
        lines = [line for line in f if not line.strip().startswith("#")]

    if not lines:
        raise ValueError(f"CSV file is empty or contains only comments: {path}")

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"CSV file has no header row: {path}")

    # Validate required columns
    required = {
        "product_id", "product_name", "price", "currency", "material",
        "care_instructions", "return_policy", "shipping_standard_days",
        "shipping_express_days", "occasion_tags", "warmth_score",
        "waterproof_score", "office_appropriate_score", "non_existent_colors",
    }
    actual = set(reader.fieldnames)
    missing = required - actual
    if missing:
        raise ValueError(
            f"Ground truth CSV is missing required columns: {sorted(missing)}\n"
            f"See ground_truth/template.csv for the expected format."
        )

    products: list[Product] = []
    for i, row in enumerate(reader, start=2):
        try:
            products.append(Product(
                product_id=row["product_id"].strip(),
                product_name=row["product_name"].strip(),
                price=_safe_float(row["price"]),
                currency=row["currency"].strip(),
                material=row["material"].strip(),
                care_instructions=row["care_instructions"].strip(),
                return_policy=row["return_policy"].strip(),
                shipping_standard_days=_safe_int(row["shipping_standard_days"]),
                shipping_express_days=_safe_int(row["shipping_express_days"]),
                hardware_color=_safe_str(row.get("hardware_color")),
                fit_type=_safe_str(row.get("fit_type")),
                dimensions_cm=_safe_str(row.get("dimensions_cm")),
                laptop_compatibility=_safe_str(row.get("laptop_compatibility")),
                sizing_notes=_safe_str(row.get("sizing_notes")),
                model_height=_safe_str(row.get("model_height")),
                model_weight=_safe_str(row.get("model_weight")),
                model_size_worn=_safe_str(row.get("model_size_worn")),
                stock=_parse_stock_columns(row),
                occasion_tags=_split_csv(row.get("occasion_tags")),
                warmth_score=_safe_float(row.get("warmth_score")),
                waterproof_score=_safe_float(row.get("waterproof_score")),
                office_appropriate_score=_safe_float(row.get("office_appropriate_score")),
                weight_grams=_safe_float(row.get("weight_grams")),
                is_lightest_in_catalog=_safe_bool(row.get("is_lightest_in_catalog")),
                is_most_packable=_safe_bool(row.get("is_most_packable")),
                catalog_rank_by_weight=_safe_int(row.get("catalog_rank_by_weight")),
                non_existent_colors=_split_csv(row.get("non_existent_colors")),
            ))
        except Exception as e:
            raise ValueError(f"Error parsing row {i} of {path}: {e}") from e

    if not products:
        raise ValueError(f"No product rows found in {path}")

    return products


def build_structured_feed(products: list[Product]) -> str:
    """Convert Product list to compact JSON feed for Agent B context injection.

    This simulates what unstructrd.brand.com would serve: clean, machine-readable
    product data with ~300 tokens per product instead of 270K tokens of HTML.
    """
    feed = []
    for p in products:
        item: dict = {
            "name": p.product_name,
            "price": p.price,
            "currency": p.currency,
            "material": p.material,
            "care": p.care_instructions,
            "return_policy": p.return_policy,
            "shipping": {
                "standard_days": p.shipping_standard_days,
                "express_days": p.shipping_express_days,
            },
            "occasion_tags": p.occasion_tags,
            "available_colors": p.available_colors,
            "available_sizes": p.available_sizes,
            "stock": [
                {"size": v.size, "color": v.color, "in_stock": v.in_stock}
                for v in p.stock
            ],
        }
        if p.hardware_color:
            item["hardware_color"] = p.hardware_color
        if p.weight_grams:
            item["weight_grams"] = p.weight_grams
        if p.fit_type:
            item["fit"] = p.fit_type
        if p.dimensions_cm:
            item["dimensions_cm"] = p.dimensions_cm
        if p.laptop_compatibility:
            item["laptop_compatibility"] = p.laptop_compatibility
        if p.sizing_notes:
            item["sizing_notes"] = p.sizing_notes
        if p.model_height or p.model_weight or p.model_size_worn:
            item["fit_model"] = {}
            if p.model_height:
                item["fit_model"]["height"] = p.model_height
            if p.model_weight:
                item["fit_model"]["weight"] = p.model_weight
            if p.model_size_worn:
                item["fit_model"]["size_worn"] = p.model_size_worn
        if p.warmth_score:
            item["warmth_score"] = p.warmth_score
        if p.waterproof_score:
            item["waterproof_score"] = p.waterproof_score
        if p.office_appropriate_score:
            item["office_appropriate_score"] = p.office_appropriate_score
        if p.occasion_tags:
            item["occasion_tags"] = p.occasion_tags
        if p.is_lightest_in_catalog:
            item["is_lightest_in_catalog"] = True
        if p.is_most_packable:
            item["is_most_packable"] = True
        if p.catalog_rank_by_weight:
            item["catalog_rank_by_weight"] = p.catalog_rank_by_weight
        feed.append(item)
    return json.dumps({"products": feed, "last_updated": "real-time"}, indent=2)
