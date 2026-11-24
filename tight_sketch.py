from dataclasses import dataclass
import hashlib
import random


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

    def estimate(self, key: object) -> int:
        for r in range(self.rows):
            q = self._hash(r, key)
            bucket = self.grid[r][q]
            if bucket.k == key:
                return bucket.c
        return 0

    def query(self, threshold: float) -> list[tuple[object, int, int]]:
        return [
            (b.k, b.c, b.a)
            for row in self.grid
            for b in row
            if b.k is not None and b.c > threshold
        ]
