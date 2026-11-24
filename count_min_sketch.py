import hashlib


class CountMinSketch:
    def __init__(self, rows: int, buckets_per_row: int, seed: int = 42):
        self.rows = rows
        self.buckets_per_row = buckets_per_row
        self.seed = seed
        self.grid = [[0] * buckets_per_row for _ in range(rows)]
        self.max_tracked = rows * buckets_per_row
        self.tracked: dict[object, int] = {}

    def _hash(self, row_idx: int, key: object) -> int:
        payload = f"{self.seed}:{row_idx}:{key}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.buckets_per_row

    def update(self, key: object) -> None:
        est = float("inf")
        for r in range(self.rows):
            q = self._hash(r, key)
            self.grid[r][q] += 1
            if self.grid[r][q] < est:
                est = self.grid[r][q]

        if key in self.tracked:
            self.tracked[key] = est
        elif len(self.tracked) < self.max_tracked:
            self.tracked[key] = est
        else:
            min_key = min(self.tracked, key=self.tracked.get)
            if est > self.tracked[min_key]:
                del self.tracked[min_key]
                self.tracked[key] = est

    def estimate(self, key: object) -> int:
        return min(
            self.grid[r][self._hash(r, key)]
            for r in range(self.rows)
        )

    def query(self, threshold: float) -> list[tuple[object, int]]:
        return [
            (k, est)
            for k, est in self.tracked.items()
            if est > threshold
        ]
