import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiff", type=Path)
    ap.add_argument("--overview", type=int, default=3, help="overview level; 3 is 1024x1024 for this tile")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with rasterio.open(args.tiff) as src:
        scale = 2 ** args.overview
        h, w = src.height // scale, src.width // scale
        raw = src.read(out_shape=(src.count, h, w), resampling=rasterio.enums.Resampling.nearest)
        raw = np.asarray(raw, dtype=np.int8).transpose(1, 2, 0)
        args.out.write_bytes(raw.tobytes(order="C"))
        meta = {"shape": list(raw.shape), "dtype": "int8", "source": str(args.tiff), "overview": args.overview}
        args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    values = raw.reshape(-1).astype(np.float32)
    np.sign(values, out=values)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        x = raw.reshape(-1).astype(np.float32)
        valid = x != -128
        x[valid] = np.sign(x[valid]) * (np.abs(x[valid]) / 127.5) ** 2
        x[~valid] = 0
        times.append(time.perf_counter() - t0)
    result = np.asarray(x, dtype=np.float32)
    args.out.with_suffix(".cpu.f32").write_bytes(result.tobytes())
    print(json.dumps({"shape": list(raw.shape), "bytes": raw.nbytes, "cpu_seconds": times, "cpu_median_seconds": float(np.median(times))}))


if __name__ == "__main__":
    main()
