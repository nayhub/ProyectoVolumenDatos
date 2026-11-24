from collections import Counter
from dataclasses import dataclass
import hashlib
import random
import pandas as pd
import pyarrow.parquet as pq


PARQUET_PATH = "yellow_tripdata_2026-09.parquet"
ZONES_CSV_PATH = "taxi_zone_lookup.csv"
ROWS = 3
BUCKETS_PER_ROW = 512
M_THRESHOLD = 10
PHI = 0.0005 
SEED = 42


@dataclass
class Bucket:
    k: object = None
    c: int = 0
    a: int = 0


class TightSketch:
    def __init__(self, rows: int, buckets_per_row: int, m_threshold: int = 10, seed: int = 42):
        self.rows = rows
        self.buckets_per_row = buckets_per_row
        self.m_threshold = m_threshold
        self.seed = seed
        self.grid = [[Bucket() for _ in range(buckets_per_row)] for _ in range(rows)]

    def _hash(self, row_idx: int, key: object) -> int:
        payload = f"{self.seed}:{row_idx}:{key}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.buckets_per_row

    def update(self, key: object) -> None:
        min_c = float("inf")
        cand_r = 0
        cand_c = 0

        for r in range(self.rows):
            q = self._hash(r, key)
            bucket = self.grid[r][q]

            if bucket.k is None or bucket.k == key:
                bucket.k = key
                bucket.c += 1
                bucket.a += 1
                return

            if bucket.c < min_c:
                min_c = bucket.c
                cand_r = r
                cand_c = q

            bucket.a = max(bucket.a - 1, 0)

        target = self.grid[cand_r][cand_c]
        decay_prob = (
            1.0 / (target.c + 1)
            if target.c < self.m_threshold
            else 1.0 / (target.c * target.a + 1)
        )

        if random.random() < decay_prob:
            target.c -= 1
            if target.c == 0:
                target.k = key
                target.c = 1

    def query(self, threshold: float) -> list[tuple[object, int, int]]:
        return [
            (b.k, b.c, b.a)
            for row in self.grid
            for b in row
            if b.k is not None and b.c > threshold
        ]


def load_zone_names(csv_path: str = "taxi_zone_lookup.csv") -> dict[int, str]:
    zones_df = pd.read_csv(csv_path).dropna(subset=["LocationID", "Zone"])
    mapping = {}
    for _, row in zones_df.iterrows():
        loc_id = int(row["LocationID"])
        borough = row.get("Borough", "")
        zone = row["Zone"]
        mapping[loc_id] = f"{zone} ({borough})" if pd.notna(borough) and borough else zone
    return mapping


def main() -> None:
    random.seed(SEED)

    table = pq.read_table(
        PARQUET_PATH,
        columns=["tpep_pickup_datetime", "PULocationID", "DOLocationID"],
    )
    df = table.to_pandas().dropna(
        subset=["tpep_pickup_datetime", "PULocationID", "DOLocationID"]
    )
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)
    df = df.sort_values("tpep_pickup_datetime")

    routes = list(zip(df["PULocationID"], df["DOLocationID"]))
    total_trips = len(routes)

    exact_counts = Counter(routes)
    threshold = PHI * total_trips
    true_heavy = {route for route, cnt in exact_counts.items() if cnt >= threshold}

    sketch = TightSketch(
        rows=ROWS,
        buckets_per_row=BUCKETS_PER_ROW,
        m_threshold=M_THRESHOLD,
        seed=SEED,
    )
    for route in routes:
        sketch.update(route)

    estimated = sketch.query(threshold)
    pred_heavy = {route for route, _, _ in estimated}

    tp = len(pred_heavy & true_heavy)
    fp = len(pred_heavy - true_heavy)
    fn = len(true_heavy - pred_heavy)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"Total de viajes procesados: {total_trips}")
    print(f"Rutas distintas reales: {len(exact_counts)}")
    print(f"Umbral de frecuencia (c > {threshold:.1f}):")
    print(f"Rutas frecuentes reales: {len(true_heavy)} | Detectadas por sketch: {len(pred_heavy)}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    zone_names = load_zone_names(ZONES_CSV_PATH)

    def fmt_route(route: tuple[int, int]) -> str:
        pu, do = route
        pu_name = zone_names.get(pu, f"ID {pu}")
        do_name = zone_names.get(do, f"ID {do}")
        return f"{pu_name} [{pu}] -> {do_name} [{do}]"

    print("\nTop 10 rutas detectadas por Tight-Sketch:")
    for route, c_est, a_est in sorted(estimated, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fmt_route(route)}: est={c_est}, real={exact_counts[route]}, fuerza={a_est}")

    missed_routes = sorted(
        true_heavy - pred_heavy,
        key=lambda route: exact_counts[route],
        reverse=True,
    )
    print(f"\nRutas frecuentes omitidas ({len(missed_routes)}) - Top 10:")
    for route in missed_routes[:10]:
        print(f"  {fmt_route(route)}: real={exact_counts[route]}")


if __name__ == "__main__":
    main()
