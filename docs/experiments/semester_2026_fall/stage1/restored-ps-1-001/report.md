# restored-ps-1-001

Status: passed

Single-process wall time includes model/data construction and 20-step history (or configured steps); PS elapsed time also includes spawn, IPC and cleanup. These are one-run demonstration timings, not a scaling benchmark.

Verified every update against single-process FP64 MBGD, including step 1 and step 20. Maximum parameter absolute difference: 0.

Workers never update weights; server weights local mean gradients by actual shard sample counts. Launcher cleanup and individual exit codes are in results.json.

See config.json, results.json and timings.csv for raw evidence.
