import time
from pathlib import Path
import numpy as np

base = Path(__file__).parent
raw = np.fromfile(base / "overview_1024x1024x64.raw", dtype=np.int8).reshape(-1, 64)
rng = np.random.default_rng(42)
query = rng.normal(size=64).astype(np.float32)
query /= np.linalg.norm(query)
query.tofile(base / "query.f32")

times = []
for _ in range(5):
    t0 = time.perf_counter()
    x = raw.astype(np.float32)
    valid = x != -128
    mag = np.abs(x) / 127.5
    x = np.sign(x) * mag * mag
    x[~valid] = 0
    result = (x @ query) / np.maximum(np.linalg.norm(x, axis=1), 1e-30)
    result[~valid.any(axis=1)] = 0
    times.append(time.perf_counter() - t0)
result.astype(np.float32).tofile(base / "overview_1024x1024x64.cpu_cosine.f32")
print({"pixels": int(raw.shape[0]), "cpu_seconds": times, "cpu_median_seconds": float(np.median(times))})
