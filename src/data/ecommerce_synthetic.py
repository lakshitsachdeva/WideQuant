"""Synthetic e-commerce dataset builder for WideQuant arithmetic retrieval."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

TEST_CATEGORY_COUNTS = {"phone": 2500, "laptop": 2500, "price": 5000}
DEV_CATEGORY_COUNTS = {"phone": 2500, "laptop": 2500, "price": 5000}


@dataclass(slots=True)
class EcommerceProduct:
    """Structured synthetic product used to generate WideQuant documents and queries."""

    product_id: str
    category: str
    name: str
    total_value: float
    atomic_unit: str
    screen_on_hours: float | None = None
    standby_hours: float | None = None
    ssd_gb: float | None = None
    hdd_gb: float | None = None
    original_price: float | None = None
    discount_pct: float | None = None


def _format_number(value: float, decimals: int = 2) -> str:
    """Render floats compactly for natural text."""
    rendered = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _variant_doc_id(product_id: str, doc_type: str) -> str:
    """Create a stable document id for one product/type pair."""
    return f"{product_id}__{doc_type}"


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_qrels(queries: Iterable[dict[str, Any]], path: Path) -> None:
    """Write TREC-style qrels for all query/document relevance pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            for doc_id in query["relevant_doc_ids"]:
                handle.write(f"{query['query_id']}\t0\t{doc_id}\t1\n")


def _phone_name(rng: random.Random) -> str:
    """Generate a phone product name."""
    brands = ["Aster", "Nova", "Vertex", "Orbit", "Pulse", "Luma"]
    series = ["Air", "Pro", "Max", "Lite", "Ultra", "Edge"]
    return f"{rng.choice(brands)} Phone {rng.choice(series)}"


def _laptop_name(rng: random.Random) -> str:
    """Generate a laptop product name."""
    brands = ["Meridian", "Atlas", "Nimbus", "Cobalt", "Aurora", "Vector"]
    models = ["13", "14", "15", "16", "Studio", "Book"]
    return f"{rng.choice(brands)} Laptop {rng.choice(models)}"


def _price_name(rng: random.Random) -> str:
    """Generate a generic consumer product name."""
    categories = ["Headphones", "Monitor", "Chair", "Tablet", "Speaker", "Camera"]
    adjectives = ["Prime", "Flex", "Zen", "Core", "Shift", "Spark"]
    return f"{rng.choice(adjectives)} {rng.choice(categories)}"


def generate_phone_products(n_products: int = 5000, seed: int = 42) -> list[EcommerceProduct]:
    """Generate synthetic phones with atomic and additive battery specs."""
    rng = random.Random(seed)
    products: list[EcommerceProduct] = []
    for idx in range(n_products):
        screen_on = round(rng.uniform(4.5, 12.5), 1)
        standby = round(rng.uniform(36.0, 120.0), 1)
        total_battery = round(screen_on + standby / 12.0, 1)
        products.append(
            EcommerceProduct(
                product_id=f"phone_{idx:05d}",
                category="phone",
                name=_phone_name(rng),
                total_value=total_battery,
                atomic_unit="hours",
                screen_on_hours=screen_on,
                standby_hours=standby,
            )
        )
    return products


def generate_laptop_products(n_products: int = 5000, seed: int = 42) -> list[EcommerceProduct]:
    """Generate synthetic laptops with additive storage specs."""
    rng = random.Random(seed + 1)
    ssd_choices = [128, 256, 512, 1024]
    hdd_choices = [256, 512, 1000, 2000]
    products: list[EcommerceProduct] = []
    for idx in range(n_products):
        ssd = float(rng.choice(ssd_choices))
        hdd = float(rng.choice(hdd_choices))
        total_storage = ssd + hdd
        products.append(
            EcommerceProduct(
                product_id=f"laptop_{idx:05d}",
                category="laptop",
                name=_laptop_name(rng),
                total_value=total_storage,
                atomic_unit="GB",
                ssd_gb=ssd,
                hdd_gb=hdd,
            )
        )
    return products


def generate_price_products(n_products: int = 10000, seed: int = 42) -> list[EcommerceProduct]:
    """Generate synthetic products with atomic price and discount decomposition."""
    rng = random.Random(seed + 2)
    discount_choices = [5, 10, 12, 15, 20, 25, 30, 35, 40]
    products: list[EcommerceProduct] = []
    for idx in range(n_products):
        original_price = float(rng.randrange(120, 2500, 10))
        discount_pct = float(rng.choice(discount_choices))
        final_price = round(original_price * (1.0 - discount_pct / 100.0), 2)
        products.append(
            EcommerceProduct(
                product_id=f"price_{idx:05d}",
                category="price",
                name=_price_name(rng),
                total_value=final_price,
                atomic_unit="$",
                original_price=original_price,
                discount_pct=discount_pct,
            )
        )
    return products


def generate_atomic_doc(product: EcommerceProduct) -> str:
    """Generate the atomic document variant for a product."""
    if product.category == "phone":
        return f"{product.name}. Battery life: {_format_number(product.total_value, 1)} hours."
    if product.category == "laptop":
        return f"{product.name}. Total storage: {_format_number(product.total_value)} GB."
    if product.category == "price":
        return f"{product.name}. Price: ${_format_number(product.total_value)}."
    raise ValueError(f"Unsupported product category: {product.category}")


def generate_decomposed_doc(product: EcommerceProduct) -> tuple[str, str]:
    """Generate the arithmetic document variant and its arithmetic type."""
    if product.category == "phone":
        text = (
            f"{product.name}. Screen-on time: {_format_number(product.screen_on_hours or 0.0, 1)} hours. "
            f"Standby time: {_format_number(product.standby_hours or 0.0, 1)} hours."
        )
        return text, "typeA"
    if product.category == "laptop":
        text = (
            f"{product.name}. SSD: {_format_number(product.ssd_gb or 0.0)} GB. "
            f"HDD: {_format_number(product.hdd_gb or 0.0)} GB."
        )
        return text, "typeA"
    if product.category == "price":
        text = (
            f"{product.name}. Original price: ${_format_number(product.original_price or 0.0)}. "
            f"Discount: {_format_number(product.discount_pct or 0.0)}%."
        )
        return text, "typeB"
    raise ValueError(f"Unsupported product category: {product.category}")


def build_documents(products: list[EcommerceProduct]) -> list[dict[str, Any]]:
    """Build atomic and arithmetic document variants for all products."""
    documents: list[dict[str, Any]] = []
    for product in products:
        atomic_doc = {
            "doc_id": _variant_doc_id(product.product_id, "atomic"),
            "product_id": product.product_id,
            "category": product.category,
            "doc_type": "atomic",
            "text": generate_atomic_doc(product),
            **asdict(product),
        }
        decomposed_text, arith_type = generate_decomposed_doc(product)
        arithmetic_doc = {
            "doc_id": _variant_doc_id(product.product_id, arith_type),
            "product_id": product.product_id,
            "category": product.category,
            "doc_type": arith_type,
            "text": decomposed_text,
            **asdict(product),
        }
        documents.extend([atomic_doc, arithmetic_doc])
    return documents


def _thresholds_for_product(product: EcommerceProduct, queries_per_product: int) -> list[float]:
    """Generate query thresholds that keep the product relevant."""
    if product.category in {"phone", "laptop"}:
        ratios = [0.72, 0.84, 0.94]
    else:
        ratios = [1.05, 1.15, 1.30]
    return [product.total_value * ratio for ratio in ratios[:queries_per_product]]


def generate_queries(products: list[EcommerceProduct], queries_per_product: int = 3) -> list[dict[str, Any]]:
    """Generate three retrieval queries per product."""
    if queries_per_product < 1 or queries_per_product > 3:
        raise ValueError("queries_per_product must be between 1 and 3 inclusive.")

    queries: list[dict[str, Any]] = []
    for product in products:
        relevant_types = ["atomic", "typeA"] if product.category in {"phone", "laptop"} else ["atomic", "typeB"]
        relevant_doc_ids = [_variant_doc_id(product.product_id, doc_type) for doc_type in relevant_types]
        thresholds = _thresholds_for_product(product, queries_per_product)

        for local_idx, threshold in enumerate(thresholds):
            if product.category == "phone":
                query_text = f"phone with battery life greater than {_format_number(threshold, 1)} hours"
                operator = "gt"
                query_type = "typeA"
            elif product.category == "laptop":
                query_text = f"laptop with total storage above {_format_number(threshold)} GB"
                operator = "gt"
                query_type = "typeA"
            else:
                query_text = f"product available for under ${_format_number(threshold)}"
                operator = "lt"
                query_type = "typeB"

            queries.append(
                {
                    "query_id": f"ecommerce_{product.product_id}_{local_idx}",
                    "query_text": query_text,
                    "product_id": product.product_id,
                    "category": product.category,
                    "query_type": query_type,
                    "relevant_doc_ids": relevant_doc_ids,
                    "relevant_types": relevant_types,
                    "threshold_value": float(round(threshold, 4)),
                    "operator": operator,
                }
            )
    return queries


def _assign_splits(queries: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Assign train/dev/test splits with a 10k stratified test set."""
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {"phone": [], "laptop": [], "price": []}
    for query in queries:
        pools[str(query["category"])].append(query)
    for pool in pools.values():
        rng.shuffle(pool)
    original_sizes = {category: len(pool) for category, pool in pools.items()}

    def split_count(category: str, target_count: int) -> int:
        if len(queries) >= 60000:
            return target_count
        scaled = int(round(original_sizes[category] / 6.0))
        return min(len(pools[category]), max(1, scaled))

    split_queries: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}

    for split_name, target_map in [("dev", DEV_CATEGORY_COUNTS), ("test", TEST_CATEGORY_COUNTS)]:
        for category, target_count in target_map.items():
            count = split_count(category, target_count)
            chosen = pools[category][:count]
            pools[category] = pools[category][count:]
            for query in chosen:
                query["split"] = split_name
                split_queries[split_name].append(query)

    for category_pool in pools.values():
        for query in category_pool:
            query["split"] = "train"
            split_queries["train"].append(query)

    return split_queries["train"] + split_queries["dev"] + split_queries["test"]


def build_dataset(
    output_dir: str = "data/ecommerce/",
    n_phones: int = 5000,
    n_laptops: int = 5000,
    n_prices: int = 10000,
    queries_per_product: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Build and save the WideQuant e-commerce synthetic dataset."""
    phone_products = generate_phone_products(n_products=n_phones, seed=seed)
    laptop_products = generate_laptop_products(n_products=n_laptops, seed=seed)
    price_products = generate_price_products(n_products=n_prices, seed=seed)
    all_products = phone_products + laptop_products + price_products

    documents = build_documents(all_products)
    queries = generate_queries(all_products, queries_per_product=queries_per_product)
    queries = _assign_splits(queries, seed=seed)

    output_path = Path(output_dir)
    documents_path = output_path / "documents.jsonl"
    queries_path = output_path / "queries.jsonl"
    qrels_path = output_path / "qrels.tsv"

    _write_jsonl(documents, documents_path)
    _write_jsonl(queries, queries_path)
    _write_qrels(queries, qrels_path)

    split_counts = {"train": 0, "dev": 0, "test": 0}
    category_counts = {"phone": 0, "laptop": 0, "price": 0}
    query_type_counts = {"typeA": 0, "typeB": 0}
    for query in queries:
        split_counts[str(query["split"])] += 1
        category_counts[str(query["category"])] += 1
        query_type_counts[str(query["query_type"])] += 1

    summary = {
        "n_products": len(all_products),
        "n_documents": len(documents),
        "n_queries": len(queries),
        "n_qrels": sum(len(query["relevant_doc_ids"]) for query in queries),
        "split_counts": split_counts,
        "category_counts": category_counts,
        "query_type_counts": query_type_counts,
        "documents_path": str(documents_path),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
    }

    print("E-commerce synthetic dataset summary:")
    print(f"- products: {summary['n_products']}")
    print(f"- documents: {summary['n_documents']}")
    print(f"- queries: {summary['n_queries']}")
    print(f"- qrels: {summary['n_qrels']}")
    print(f"- split counts: {summary['split_counts']}")
    print(f"- category counts: {summary['category_counts']}")
    print(f"- query type counts: {summary['query_type_counts']}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Build the WideQuant e-commerce synthetic dataset")
    parser.add_argument("--output_dir", type=str, default="data/ecommerce")
    parser.add_argument("--n_phones", type=int, default=5000)
    parser.add_argument("--n_laptops", type=int, default=5000)
    parser.add_argument("--n_prices", type=int, default=10000)
    parser.add_argument("--queries_per_product", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = build_dataset(
        output_dir=args.output_dir,
        n_phones=int(args.n_phones),
        n_laptops=int(args.n_laptops),
        n_prices=int(args.n_prices),
        queries_per_product=int(args.queries_per_product),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))
