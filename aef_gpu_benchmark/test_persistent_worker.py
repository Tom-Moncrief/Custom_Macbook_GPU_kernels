import subprocess
import time
from pathlib import Path

import numpy as np

base = Path(__file__).parent
binary = base / "gpu_worker"
raw = base / "overview_1024x1024x64.raw"
query = base / "query.f32"
outputs = [base / f"worker_score_{i}.f32" for i in range(5)]

worker = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
assert worker.stdout.readline().strip() == "READY"
times = []
for output in outputs:
    t0 = time.perf_counter()
    worker.stdin.write(f"{raw}\t{query}\t{output}\n")
    worker.stdin.flush()
    assert worker.stdout.readline().startswith("DONE\t")
    times.append(time.perf_counter() - t0)
worker.stdin.write("QUIT\n")
worker.stdin.flush()
return_code = worker.wait()
assert return_code == 0, return_code

reference = np.fromfile(base / "overview_1024x1024x64.cpu_cosine.f32", dtype=np.float32)
result = np.fromfile(outputs[-1], dtype=np.float32)
print({
    "jobs": len(times),
    "worker_job_seconds": times,
    "median_seconds": float(np.median(times)),
    "speedup_vs_cpu": float(0.7540989170083776 / np.median(times)),
    "max_abs_error": float(np.max(np.abs(reference - result))),
})
