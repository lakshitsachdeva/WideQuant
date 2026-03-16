"""OpenFoodFacts dataset builder for WideQuant arithmetic retrieval experiments."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

OPENFOODFACTS_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
REQUIRED_COLUMNS = ["proteins_100g", "fat_100g", "carbohydrates_100g"]
QUERY_TYPE_ORDER = ["typeA", "typeC", "typeB", "mixed", "atomic"]
SPLIT_TARGETS = {
    "dev": {"atomic": 1250, "typeA": 1250, "typeB": 1250, "typeC": 625, "mixed": 625},
    "test": {"atomic": 1250, "typeA": 1250, "typeB": 1250, "typeC": 625, "mixed": 625},
}
ALL_DOC_TYPES = ["atomic", "typeA", "typeB", "typeC"]
SPLIT_TYPE_RATIOS = {"atomic": 0.25, "typeA": 0.25, "typeB": 0.25, "typeC": 0.125, "mixed": 0.125}


@dataclass(slots=True)
class FoodProduct:
    """Container for one OpenFoodFacts product used to build documents and queries."""

    name: str
    energy_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    product_id: str


def _format_number(value: float, decimals: int = 2) -> str:
    """Format a float compactly for text generation."""
    rendered = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _safe_product_name(value: Any, fallback: str) -> str:
    """Return a usable product name, falling back to the product id when needed."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return fallback


def _infer_delimiter(csv_path: Path) -> str:
    """Infer CSV delimiter from a short sample of the decompressed file."""
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") > sample.count(",") else ","


def _load_openfoodfacts_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load the OpenFoodFacts CSV with a robust delimiter guess."""
    delimiter = _infer_delimiter(csv_path)
    return pd.read_csv(csv_path, sep=delimiter, low_memory=False)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert text-like numeric columns to floats."""
    return pd.to_numeric(series, errors="coerce")


def _normalize_products_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw OpenFoodFacts fields into the columns WideQuant needs."""
    working = df.copy()

    if "energy-kcal_100g" in working.columns:
        working["energy_kcal_100g"] = _coerce_numeric(working["energy-kcal_100g"])
    elif "energy_100g" in working.columns:
        working["energy_kcal_100g"] = _coerce_numeric(working["energy_100g"]) / 4.184
    else:
        working["energy_kcal_100g"] = pd.Series(dtype="float64")

    for column in REQUIRED_COLUMNS:
        if column not in working.columns:
            working[column] = pd.Series(dtype="float64")
        else:
            working[column] = _coerce_numeric(working[column])

    if "code" in working.columns:
        working["product_id"] = working["code"].astype(str).str.strip()
    elif "id" in working.columns:
        working["product_id"] = working["id"].astype(str).str.strip()
    else:
        working["product_id"] = working.index.astype(str)

    name_series = None
    for candidate in ["product_name", "product_name_en", "generic_name", "abbreviated_product_name"]:
        if candidate in working.columns:
            name_series = working[candidate]
            break
    if name_series is None:
        working["name"] = working["product_id"]
    else:
        working["name"] = [
            _safe_product_name(value, fallback=str(product_id))
            for value, product_id in zip(name_series, working["product_id"], strict=False)
        ]

    working["nonnull_count"] = working.notna().sum(axis=1)
    return working


def download_openfoodfacts_csv(output_dir: str) -> Path:
    """Download and decompress the OpenFoodFacts product dump into output_dir/openfoodfacts.csv."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gz_path = output_path / "openfoodfacts.csv.gz"
    csv_path = output_path / "openfoodfacts.csv"

    if not csv_path.exists():
        if not gz_path.exists():
            print(f"Downloading OpenFoodFacts dump to {gz_path} ...")
            urllib.request.urlretrieve(OPENFOODFACTS_URL, gz_path)
        print(f"Decompressing {gz_path} -> {csv_path} ...")
        with gzip.open(gz_path, "rb") as src_handle, csv_path.open("wb") as dst_handle:
            shutil.copyfileobj(src_handle, dst_handle)

    return csv_path


def filter_complete_products(df: pd.DataFrame, n: int = 10000) -> pd.DataFrame:
    """Keep complete products and return the top n with the richest field coverage."""
    working = _normalize_products_frame(df)
    complete = working.dropna(
        subset=["energy_kcal_100g", "proteins_100g", "fat_100g", "carbohydrates_100g"]
    ).copy()
    complete = complete[complete["energy_kcal_100g"] > 0.0]
    complete = complete[
        (complete["proteins_100g"] >= 0.0)
        & (complete["fat_100g"] >= 0.0)
        & (complete["carbohydrates_100g"] >= 0.0)
    ]
    complete["macro_energy_kcal"] = (
        complete["proteins_100g"] * 4.0
        + complete["fat_100g"] * 9.0
        + complete["carbohydrates_100g"] * 4.0
    )
    complete = complete[
        (complete["macro_energy_kcal"] - complete["energy_kcal_100g"]).abs() <= 5.0
    ]
    complete = complete.sort_values("nonnull_count", ascending=False).head(int(n))
    return complete.reset_index(drop=True)


def _row_to_product(row: pd.Series) -> FoodProduct:
    """Convert one normalized DataFrame row into a FoodProduct."""
    product_id = str(row["product_id"])
    return FoodProduct(
        name=_safe_product_name(row["name"], fallback=product_id),
        energy_kcal=float(row["energy_kcal_100g"]),
        protein_g=float(row["proteins_100g"]),
        fat_g=float(row["fat_100g"]),
        carbs_g=float(row["carbohydrates_100g"]),
        product_id=product_id,
    )


def _variant_doc_id(product_id: str, doc_type: str) -> str:
    """Create a stable document id for one product/type pair."""
    return f"{product_id}__{doc_type}"


def generate_atomic_doc(product: FoodProduct) -> str:
    """Generate a direct atomic nutritional document."""
    return (
        f"{product.name} - per 100g. "
        f"Energy: {_format_number(product.energy_kcal)} kcal. "
        f"Protein: {_format_number(product.protein_g)}g. "
        f"Fat: {_format_number(product.fat_g)}g. "
        f"Carbs: {_format_number(product.carbs_g)}g."
    )


def generate_typeA_doc(product: FoodProduct) -> str:
    """Generate an additive decomposition document over macro-derived calories."""
    protein_kcal = product.protein_g * 4.0
    fat_kcal = product.fat_g * 9.0
    carbs_kcal = product.carbs_g * 4.0
    return (
        f"{product.name} - energy from protein: {_format_number(protein_kcal)} kcal. "
        f"Energy from fat: {_format_number(fat_kcal)} kcal. "
        f"Energy from carbohydrates: {_format_number(carbs_kcal)} kcal."
    )


def generate_typeB_doc(product: FoodProduct) -> str:
    """Generate a ratio-style document that supports percentage reasoning."""
    protein_kcal = product.protein_g * 4.0
    return (
        f"{product.name} - protein: {_format_number(product.protein_g)}g providing "
        f"{_format_number(protein_kcal)} kcal. Total energy: {_format_number(product.energy_kcal)} kcal."
    )


def generate_typeC_doc(product: FoodProduct) -> str:
    """Generate a unit-converted document that uses kJ instead of kcal."""
    energy_kj = product.energy_kcal * 4.184
    return (
        f"{product.name} - energy: {_format_number(energy_kj)} kJ. "
        f"Protein: {_format_number(product.protein_g)}g. "
        f"Fat: {_format_number(product.fat_g)}g."
    )


def _build_documents(products: Iterable[FoodProduct]) -> list[dict[str, Any]]:
    """Create all four document variants for each product."""
    documents: list[dict[str, Any]] = []
    for product in products:
        protein_kcal = product.protein_g * 4.0
        fat_kcal = product.fat_g * 9.0
        carbs_kcal = product.carbs_g * 4.0
        energy_kj = product.energy_kcal * 4.184
        texts = {
            "atomic": generate_atomic_doc(product),
            "typeA": generate_typeA_doc(product),
            "typeB": generate_typeB_doc(product),
            "typeC": generate_typeC_doc(product),
        }
        for doc_type, text in texts.items():
            documents.append(
                {
                    "doc_id": _variant_doc_id(product.product_id, doc_type),
                    "product_id": product.product_id,
                    "doc_type": doc_type,
                    "text": text,
                    "name": product.name,
                    "energy_kcal": float(product.energy_kcal),
                    "energy_kj": float(energy_kj),
                    "protein_g": float(product.protein_g),
                    "fat_g": float(product.fat_g),
                    "carbs_g": float(product.carbs_g),
                    "protein_kcal": float(protein_kcal),
                    "fat_kcal": float(fat_kcal),
                    "carbs_kcal": float(carbs_kcal),
                }
            )
    return documents


def _query_relevant_types(query_type: str) -> list[str]:
    """Return the document type metadata associated with an intended reasoning type."""
    mapping = {
        "atomic": ["atomic"],
        "typeA": ["typeA"],
        "typeB": ["typeB"],
        "typeC": ["typeC"],
        "mixed": ["atomic", "typeA", "typeB", "typeC"],
    }
    return mapping[query_type]


def generate_queries(products: list[FoodProduct], queries_per_product: int = 5) -> list[dict[str, Any]]:
    """Generate five WideQuant retrieval queries per product."""
    if queries_per_product < 1 or queries_per_product > 5:
        raise ValueError("queries_per_product must be between 1 and 5 inclusive.")

    queries: list[dict[str, Any]] = []
    for product in products:
        relevant_doc_ids = [_variant_doc_id(product.product_id, doc_type) for doc_type in ALL_DOC_TYPES]
        energy_gt = 0.8 * product.energy_kcal
        low_cal = 1.2 * product.energy_kcal
        protein_pct = (product.protein_g * 4.0 / product.energy_kcal * 100.0) if product.energy_kcal > 0 else 0.0
        protein_pct_thresh = max(1.0, min(95.0, protein_pct * 0.9))
        energy_and_thresh = 0.9 * product.energy_kcal
        protein_and_thresh = 0.8 * product.protein_g
        fat_thresh = 0.8 * product.fat_g

        product_queries = [
            {
                "query_text": f"food with energy content greater than {_format_number(energy_gt)} kcal",
                "relevant_product_ids": relevant_doc_ids,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_types": _query_relevant_types("typeA"),
                "threshold_value": float(energy_gt),
                "operator": "gt",
                "query_type": "typeA",
                "product_id": product.product_id,
            },
            {
                "query_text": f"low calorie food under {_format_number(low_cal)} kcal",
                "relevant_product_ids": relevant_doc_ids,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_types": _query_relevant_types("typeC"),
                "threshold_value": float(low_cal),
                "operator": "lt",
                "query_type": "typeC",
                "product_id": product.product_id,
            },
            {
                "query_text": (
                    "high protein food where protein exceeds "
                    f"{_format_number(protein_pct_thresh)}% of total calories"
                ),
                "relevant_product_ids": relevant_doc_ids,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_types": _query_relevant_types("typeB"),
                "threshold_value": float(protein_pct_thresh),
                "operator": "gt",
                "query_type": "typeB",
                "product_id": product.product_id,
            },
            {
                "query_text": (
                    f"food with energy above {_format_number(energy_and_thresh)} kcal "
                    f"and protein above {_format_number(protein_and_thresh)}g"
                ),
                "relevant_product_ids": relevant_doc_ids,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_types": _query_relevant_types("mixed"),
                "threshold_value": {
                    "energy_kcal": float(energy_and_thresh),
                    "protein_g": float(protein_and_thresh),
                },
                "operator": "and",
                "query_type": "mixed",
                "product_id": product.product_id,
            },
            {
                "query_text": f"food with fat content greater than {_format_number(fat_thresh)}g per 100g",
                "relevant_product_ids": relevant_doc_ids,
                "relevant_doc_ids": relevant_doc_ids,
                "relevant_types": _query_relevant_types("atomic"),
                "threshold_value": float(fat_thresh),
                "operator": "gt",
                "query_type": "atomic",
                "product_id": product.product_id,
            },
        ]
        queries.extend(product_queries[:queries_per_product])

    for idx, query in enumerate(queries):
        query["query_id"] = f"openfoodfacts_q_{idx:06d}"
    return queries


def _assign_splits(queries: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Assign train/dev/test splits with the requested type-stratified dev/test sets."""
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in QUERY_TYPE_ORDER}
    for query in queries:
        pools[str(query["query_type"])].append(query)
    for pool in pools.values():
        rng.shuffle(pool)

    split_to_queries: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    consumed_ids: set[str] = set()
    total_queries = len(queries)

    def compute_target_counts(split_name: str) -> dict[str, int]:
        if total_queries >= 50000:
            return dict(SPLIT_TARGETS[split_name])

        split_total = max(1, int(round(total_queries * 0.10)))
        raw_counts = {
            query_type: split_total * ratio for query_type, ratio in SPLIT_TYPE_RATIOS.items()
        }
        counts = {query_type: int(value) for query_type, value in raw_counts.items()}
        assigned = sum(counts.values())
        remainders = sorted(
            ((raw_counts[query_type] - counts[query_type], query_type) for query_type in QUERY_TYPE_ORDER),
            reverse=True,
        )
        for _, query_type in remainders:
            if assigned >= split_total:
                break
            counts[query_type] += 1
            assigned += 1
        return counts

    for split_name in ["dev", "test"]:
        for query_type, count in compute_target_counts(split_name).items():
            pool = pools[query_type]
            if len(pool) < count:
                raise ValueError(
                    f"Not enough queries for split={split_name}, query_type={query_type}: "
                    f"have {len(pool)}, need {count}"
                )
            chosen = pool[:count]
            pools[query_type] = pool[count:]
            for query in chosen:
                query["split"] = split_name
                split_to_queries[split_name].append(query)
                consumed_ids.add(str(query["query_id"]))

    for query_type in QUERY_TYPE_ORDER:
        for query in pools[query_type]:
            if str(query["query_id"]) in consumed_ids:
                continue
            query["split"] = "train"
            split_to_queries["train"].append(query)

    return split_to_queries["train"] + split_to_queries["dev"] + split_to_queries["test"]


def _write_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write a sequence of dict rows to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_qrels(queries: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write TREC-style qrels for all relevant document ids."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            query_id = str(query["query_id"])
            for doc_id in query["relevant_doc_ids"]:
                handle.write(f"{query_id}\t0\t{doc_id}\t1\n")


def build_dataset(n_products: int = 10000, output_dir: str = "data/openfoodfacts/") -> dict[str, Any]:
    """Build the OpenFoodFacts WideQuant dataset and save corpus, queries, and qrels."""
    output_path = Path(output_dir)
    csv_path = download_openfoodfacts_csv(str(output_path))
    raw_df = _load_openfoodfacts_dataframe(csv_path)
    filtered_df = filter_complete_products(raw_df, n=n_products)
    products = [_row_to_product(row) for _, row in filtered_df.iterrows()]

    documents = _build_documents(products)
    queries = generate_queries(products, queries_per_product=5)
    queries = _assign_splits(queries, seed=42)

    documents_path = output_path / "documents.jsonl"
    queries_path = output_path / "queries.jsonl"
    qrels_path = output_path / "qrels.tsv"
    _write_jsonl(documents, documents_path)
    _write_jsonl(queries, queries_path)
    _write_qrels(queries, qrels_path)

    split_counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    type_counts: dict[str, int] = {key: 0 for key in QUERY_TYPE_ORDER}
    for query in queries:
        split_counts[str(query["split"])] += 1
        type_counts[str(query["query_type"])] += 1

    summary = {
        "n_products": len(products),
        "n_documents": len(documents),
        "n_queries": len(queries),
        "n_qrels": sum(len(query["relevant_doc_ids"]) for query in queries),
        "split_counts": split_counts,
        "query_type_counts": type_counts,
        "documents_path": str(documents_path),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
    }

    print("OpenFoodFacts dataset summary:")
    print(f"- products: {summary['n_products']}")
    print(f"- documents: {summary['n_documents']}")
    print(f"- queries: {summary['n_queries']}")
    print(f"- qrels: {summary['n_qrels']}")
    print(f"- split counts: {summary['split_counts']}")
    print(f"- query type counts: {summary['query_type_counts']}")
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_dataset(output_dir: str) -> bool:
    """Verify that additive and conversion documents reconstruct the atomic energy values."""
    documents_path = Path(output_dir) / "documents.jsonl"
    if not documents_path.exists():
        raise FileNotFoundError(
            f"Missing {documents_path}. Run build_dataset() before verify_dataset()."
        )

    documents = _load_jsonl(documents_path)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for doc in documents:
        grouped.setdefault(str(doc["product_id"]), {})[str(doc["doc_type"])] = doc

    eligible_ids = [
        product_id
        for product_id, docs in grouped.items()
        if {"atomic", "typeA", "typeC"}.issubset(docs.keys())
    ]
    if len(eligible_ids) < 50:
        raise ValueError(f"Need at least 50 products to verify, found {len(eligible_ids)}")

    rng = random.Random(42)
    sample_ids = rng.sample(eligible_ids, 50)
    all_pass = True

    for idx, product_id in enumerate(sample_ids, start=1):
        atomic = grouped[product_id]["atomic"]
        type_a = grouped[product_id]["typeA"]
        type_c = grouped[product_id]["typeC"]

        target_kcal = float(atomic["energy_kcal"])
        additive_kcal = (
            float(type_a["protein_kcal"]) + float(type_a["fat_kcal"]) + float(type_a["carbs_kcal"])
        )
        converted_kcal = float(type_c["energy_kj"]) / 4.184

        type_a_pass = abs(additive_kcal - target_kcal) <= 5.0
        type_c_pass = abs(converted_kcal - target_kcal) <= 2.0
        sample_pass = type_a_pass and type_c_pass
        all_pass = all_pass and sample_pass

        print(
            f"[{idx:02d}/50] {product_id} | "
            f"TypeA {'PASS' if type_a_pass else 'FAIL'} "
            f"(sum={additive_kcal:.2f}, target={target_kcal:.2f}) | "
            f"TypeC {'PASS' if type_c_pass else 'FAIL'} "
            f"(kj->kcal={converted_kcal:.2f}, target={target_kcal:.2f})"
        )

    print(f"OVERALL {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for build/verify runs."""
    parser = argparse.ArgumentParser(description="Build and verify the WideQuant OpenFoodFacts dataset")
    parser.add_argument("--output_dir", type=str, default="data/openfoodfacts")
    parser.add_argument("--n_products", type=int, default=10000)
    parser.add_argument("--verify_only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_only:
        verify_dataset(args.output_dir)
    else:
        summary = build_dataset(n_products=int(args.n_products), output_dir=args.output_dir)
        print(json.dumps(summary, indent=2))
