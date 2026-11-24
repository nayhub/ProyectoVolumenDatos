from collections import Counter
import random
import time

from count_min_sketch import CountMinSketch
from data_utils import evaluate, load_routes, load_zone_names
from tight_sketch import TightSketch


PARQUET_PATH = "yellow_tripdata_2026-09.parquet"
ZONES_CSV_PATH = "taxi_zone_lookup.csv"
ROWS = 3
BUCKETS_PER_ROW = 512
M_THRESHOLD = 10
PHI = 0.0005  
SEED = 42


def main() -> None:
    random.seed(SEED)

    routes = load_routes(PARQUET_PATH)
    total_trips = len(routes)

    exact_counts = Counter(routes)
    threshold = PHI * total_trips
    true_heavy = {route for route, cnt in exact_counts.items() if cnt >= threshold}

    t0 = time.perf_counter()
    ts = TightSketch(rows=ROWS, buckets_per_row=BUCKETS_PER_ROW, m_threshold=M_THRESHOLD, seed=SEED)
    for route in routes:
        ts.update(route)
    ts_time = time.perf_counter() - t0
    ts_pred = {route for route, _, _ in ts.query(threshold)}
    ts_metrics = evaluate(ts_pred, true_heavy, exact_counts, ts.estimate)

    t0 = time.perf_counter()
    cms = CountMinSketch(rows=ROWS, buckets_per_row=BUCKETS_PER_ROW, seed=SEED)
    for route in routes:
        cms.update(route)
    cms_time = time.perf_counter() - t0
    cms_pred = {route for route, _ in cms.query(threshold)}
    cms_metrics = evaluate(cms_pred, true_heavy, exact_counts, cms.estimate)

    print(f"Viajes procesados: {total_trips} | Rutas distintas: {len(exact_counts)}")
    print(f"Umbral (c > {threshold:.1f}) | Rutas frecuentes reales: {len(true_heavy)}")
    print(f"Grilla: {ROWS} x {BUCKETS_PER_ROW} ({ROWS * BUCKETS_PER_ROW} buckets)\n")

    header = f"{'Algoritmo':<18} {'Detect':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8} {'F1':>8} {'ARE':>8} {'Tiempo':>8}"
    print(header)
    print("-" * len(header))
    for name, m, elapsed in [("Tight-Sketch", ts_metrics, ts_time), ("Count-Min Sketch", cms_metrics, cms_time)]:
        print(
            f"{name:<18} {m['detected']:>7} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['are']:>8.4f} {elapsed:>7.1f}s"
        )

    zone_names = load_zone_names(ZONES_CSV_PATH)

    def fmt_route(route: tuple[int, int]) -> str:
        pu, do = route
        return f"{zone_names.get(pu, f'ID {pu}')} [{pu}] -> {zone_names.get(do, f'ID {do}')} [{do}]"

    print("\nComparación de conteos en Top 10 rutas reales:")
    for route, real_cnt in exact_counts.most_common(10):
        print(
            f"  {fmt_route(route)}: "
            f"Real={real_cnt} | Tight={ts.estimate(route)} | CMS={cms.estimate(route)}"
        )


if __name__ == "__main__":
    main()
