# correctness-final-002

Status: passed

All operands are identical FP32 fixtures; errors use independent FP64 loop references. Max relative error uses denominator max(abs(reference), 1e-6).

GPU event intervals cover the whole operator call and may include GPU idle time while Python submits work. These are not kernel-only measurements. Profiler timings are excluded.

See config.json, results.json and timings.csv for raw evidence.
