import time
from pathlib import Path
import joblib
import numpy as np

base = Path(__file__).parent
x = np.fromfile(base / "overview_1024x1024x64.cpu.f32", dtype=np.float32).reshape(-1, 64)
results = {}
for name in ["regression", "classification"]:
    model = joblib.load(base / f"rf_{name}.joblib")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        pred = model.predict(x)
        times.append(time.perf_counter() - t0)
    pred.astype(np.float32).tofile(base / f"rf_{name}.cpu.timed.f32")
    results[name] = {"cpu_median_seconds": float(np.median(times)), "cpu_seconds": times, "trees": len(model.estimators_)}
print(results)
