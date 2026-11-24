from collections import Counter
import pandas as pd
import pyarrow.parquet as pq


def load_routes(parquet_path: str) -> list[tuple[int, int]]:
    table = pq.read_table(
        parquet_path,
        columns=["tpep_pickup_datetime", "PULocationID", "DOLocationID"],
    )
    df = table.to_pandas().dropna(
        subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"]
    )
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)
    df = df.sort_values("tpep_pickup_datetime")
    return list(zip(df["PULocationID"], df["DOLocationID"]))


def load_zone_names(csv_path: str = "taxi_zone_lookup.csv") -> dict[int, str]:
    zones_df = pd.read_csv(csv_path).dropna(subset=["LocationID", "Zone"])
    mapping = {}
    for _, row in zones_df.iterrows():
        loc_id = int(row["LocationID"])
        borough = row.get("Borough", "")
        zone = row["Zone"]
        mapping[loc_id] = f"{zone} ({borough})" if pd.notna(borough) and borough else zone
    return mapping


def evaluate(pred_heavy: set, true_heavy: set, exact_counts: Counter, est_fn) -> dict[str, float]:
    tp = len(pred_heavy & true_heavy)
    fp = len(pred_heavy - true_heavy)
    fn = len(true_heavy - pred_heavy)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    are = (
        sum(abs(est_fn(r) - exact_counts[r]) / exact_counts[r] for r in true_heavy) / len(true_heavy)
        if true_heavy
        else 0.0
    )
    return {
        "detected": len(pred_heavy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "are": are,
    }
