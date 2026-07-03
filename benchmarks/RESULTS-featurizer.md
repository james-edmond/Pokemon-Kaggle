# Featurizer throughput benchmark

Date: 2026-07-03

## Hardware / platform

| Item | Value |
|---|---|
| CPU | Intel Core i5-6600K @ 3.50 GHz (Skylake, 2015), 4 cores / 4 threads, no SMT, no E-cores (homogeneous) |
| RAM | 16 GiB |
| OS | Windows 10 Pro 10.0.19045, 64-bit |
| Python | 3.14.6 (MSC v.1944, AMD64) |

## Results

Benchmark run: `python benchmarks/bench_featurizer.py`

```
games=996 selections=49238
featurize+encode: mean 254 us, median 180 us, p95 425 us
```

The featurizer throughput (254 µs mean per featurize_state + encode_select call) is well within acceptable bounds for the training pipeline budget.
